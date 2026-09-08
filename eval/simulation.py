"""
The shared simulation layer: one function that runs one hour of traffic under
one controller and returns comparable numbers.

Everything else in the project - the fixed-time baseline, the DQN evaluation,
the robustness test - calls `run_episode()` from here.  Keeping a single code
path is what makes the comparison trustworthy: the two controllers are not
just measured with the same *metrics*, they are measured by the same *code*,
on the same vehicles, with the same yellow times and the same episode length.
The only thing that differs is which object picks the next signal phase.

WHAT WE MEASURE AND WHY
-----------------------
mean_waiting_time_s
    Average, over every vehicle that entered the network, of the seconds it
    spent stopped in a queue.  This is the headline number: it is what a
    driver actually experiences, and it is what the agent is rewarded for
    reducing.

mean_queue_length_veh
    Time-averaged number of stopped vehicles across the four approaches.  This
    is what the *road* experiences.  It matters separately from waiting time
    because a long queue can block upstream junctions even if each individual
    driver's delay looks acceptable.

throughput_veh
    Vehicles that completed their trip within the hour.  A controller can
    always make waiting time look good by refusing to let cars in, so waiting
    time is only meaningful when read next to throughput.

mean_time_loss_s
    Seconds lost compared to driving the same route at the free-flow speed
    with no signal at all.  Slightly fairer than raw waiting time because it
    also counts slow crawling, not just full stops.

vehicles_not_inserted
    Vehicles still queued to enter the network when the hour ended, because the
    approach they need was full.  This is the honesty check: it must be near
    zero, otherwise congestion is being hidden outside the simulation rather
    than measured inside it.

    Two things to be precise about.  It is a *snapshot* taken at the final
    step, not a running count over the hour - the two nearly coincide because
    demand stops when the episode does, so a vehicle still in that queue is one
    that never entered at all.  And a value of -1 means the measurement itself
    failed; it does not mean zero.  The distinction matters because zero is the
    reading the report relies on, so a failure must never be able to imitate it.

A NOTE ON SURVIVORSHIP BIAS
---------------------------
SUMO's per-vehicle trip report normally only records vehicles that *finished*
their journey.  Under heavy congestion the worst-delayed vehicles are exactly
the ones still stuck in a queue when the hour ends, so they would be silently
dropped and a bad controller would score better than it deserves.  We pass
`--tripinfo-output.write-unfinished` so those vehicles are included too.
"""

from __future__ import annotations

import contextlib
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sumo_setup  # noqa: E402,F401  (configures SUMO paths before any SUMO import)
import traci  # noqa: E402  (importable only after sumo_setup has run)

NET_FILE = ROOT / "networks" / "single_intersection.net.xml"
ROUTES_DIR = ROOT / "networks" / "routes"
TRIPINFO_DIR = ROOT / "results" / "tripinfo"

# ---------------------------------------------------------------------------
# Simulation constants - identical for every controller and every experiment.
# ---------------------------------------------------------------------------

# One simulated hour per episode.
#
# Why an hour: the intersection needs a few minutes to fill up from empty
# before the queues reach a steady state, so a short episode would mostly
# measure the warm-up. An hour gives ~1500-2000 completed trips, enough for the
# averages to be stable, while still running in seconds on a laptop.
EPISODE_SECONDS = 3600

# The agent makes a decision every 5 simulated seconds.
#
# Why 5 s: fast enough to react to a platoon of cars arriving, slow enough that
# one decision has a visible effect before the next is taken. It also gives
# 3600/5 = 720 decisions per episode, a convenient amount of experience.
DELTA_TIME = 5

# Amber time when the signal changes phase. Matches the 4 s amber written into
# the fixed-time plan in networks/build_network.py, so both controllers pay
# exactly the same safety cost for switching. Making these differ would be the
# easiest way to accidentally rig the comparison.
YELLOW_TIME = 4

# A phase must stay green at least this long before the agent may abandon it.
#
# Without a minimum the agent can flicker between phases every 5 s, which looks
# great to a naive reward function (it keeps "serving" everyone) but in reality
# wastes the whole intersection on amber time and would be unsafe and illegal.
# 10 s is a standard minimum green for an urban signal.
MIN_GREEN = 10

# Number of signal-controlled incoming lanes (4 approaches x 2 lanes).  Used
# only to scale the reward into a convenient range.
N_INCOMING_LANES = 8


def negative_queue_reward(signal) -> float:
    """
    Reward = minus the average number of stopped vehicles per incoming lane.

    WHY THIS REWARD
    ---------------
    stable-baselines3 always MAXIMISES reward, and we want to MINIMISE delay,
    so the quantity has to be negated - hence "negative queue".

    Adding this up over a whole episode gives minus the total vehicle-seconds
    spent stopped at the intersection.  By Little's law that is the same
    quantity as the total waiting time experienced by all drivers, so an agent
    that maximises this reward is directly minimising the headline metric we
    report.  The episode return is therefore a real performance score, which is
    exactly what we need for the training curve to mean something.

    WHY NOT sumo-rl's DEFAULT "diff-waiting-time"
    ---------------------------------------------
    That default rewards the *change* in waiting time between two steps.  The
    per-step changes telescope, so the total over an episode collapses to
    (waiting time at the start) - (waiting time at the end), which is near zero
    no matter how well or badly the agent drove the signal.  It can still train
    a policy, because discounting makes the agent prefer clearing queues early,
    but its episode return is flat noise - useless for showing that learning
    happened, which the project explicitly needs to demonstrate.

    WHY DIVIDE BY THE LANE COUNT
    ----------------------------
    Purely numerical hygiene.  Raw queue counts reach ~50 vehicles here, which
    with a discount factor of 0.95 implies Q-values around -1000; such large
    targets make the neural network's early updates unstable.  Dividing by a
    constant cannot change which policy is optimal - it just rescales every
    Q-value - and it keeps the per-step reward in a comfortable [-7, 0] band.
    The resulting number also has a plain-English meaning: "average queue per
    lane, negated".
    """
    return -signal.get_total_queued() / N_INCOMING_LANES


REWARD_FN = negative_queue_reward


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    """Metrics from a single simulated hour."""

    controller: str
    scenario: str
    seed: int

    mean_waiting_time_s: float = 0.0
    mean_queue_length_veh: float = 0.0
    max_queue_length_veh: float = 0.0
    throughput_veh: int = 0
    mean_time_loss_s: float = 0.0
    mean_trip_duration_s: float = 0.0
    mean_speed_mps: float = 0.0
    vehicles_entered: int = 0
    vehicles_not_inserted: int = 0
    total_reward: float = 0.0
    phase_switches: int = 0

    # Per-step trace, kept out of the summary table.
    trace: list = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("trace")
        return d


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class Controller:
    """
    Base class defining the small interface `run_episode` drives.

    Two attributes and two hooks, which is all the loop needs to treat a
    stopwatch, a heuristic and a neural network identically:

    uses_env_control
        True if this controller decides the phase itself. False means SUMO
        runs a static plan and our `act` return value is ignored - which is
        how the fixed-time baselines work.
    on_episode_start(env)
        Called once after the simulation starts, for controllers that need to
        configure the signal before traffic arrives.
    """

    name = "controller"
    uses_env_control = True

    def __init__(self, env=None):
        self.env = env

    def on_episode_start(self, env) -> None:
        pass

    def act(self, observation) -> int:
        raise NotImplementedError


class FixedTimeController(Controller):
    """
    The baseline: a signal that cycles through its phases on a stopwatch.

    It does not look at traffic at all.  SUMO runs the static plan compiled
    into the network file, so this class has nothing to decide - it exists so
    that the fixed-time signal and the RL agent can be driven through the
    identical `run_episode` loop, guaranteeing they are measured the same way.
    """

    name = "fixed-time"
    uses_env_control = False  # hand control to SUMO's own program

    def act(self, observation) -> int:
        return 0  # ignored


class PolicyController(Controller):
    """Wraps a trained stable-baselines3 model so it looks like a controller."""

    uses_env_control = True

    def __init__(self, model, name: str, env=None):
        super().__init__(env)
        self._model = model
        self.name = name

    def act(self, observation) -> int:
        # deterministic=True turns off the exploration noise used during
        # training: at evaluation time we want the policy's best guess, not a
        # random sample from it.
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)


class GreedyQueueController(Controller):
    """
    A simple traffic-responsive reference controller (not a learned one).

    It always switches to whichever phase currently has the most stopped
    vehicles waiting for it.  This is included because "RL beats a signal that
    ignores traffic entirely" is a weak claim - a five-line heuristic can do
    that too.  Showing where the DQN sits relative to this heuristic is a much
    more informative result, in either direction.
    """

    name = "greedy-queue"
    uses_env_control = True

    def __init__(self, env=None):
        super().__init__(env)
        self._phase_lane_slots = None

    def on_episode_start(self, env) -> None:
        self.env = env
        # The lane/phase mapping is read from the live simulation, so it has to
        # be rebuilt each episode rather than cached across them.
        self._phase_lane_slots = None

    def _build_phase_map(self, signal) -> list:
        """
        For each phase, list the positions in `signal.lanes` it would release.

        A phase is a string of lights, one per "link index". SUMO can tell us
        which incoming lane each link index belongs to, so we translate the
        light string into a set of lanes, and then into indices into the queue
        vector that `get_lanes_queue()` returns.
        """
        controlled = self.env.sumo.trafficlight.getControlledLinks(signal.id)
        lane_of_link = {}
        for index, link_group in enumerate(controlled):
            if link_group:
                lane_of_link[index] = link_group[0][0]  # (inLane, outLane, via)

        slots = []
        for phase in signal.green_phases:
            released = set()
            for index, colour in enumerate(phase.state):
                if colour in "Gg" and index in lane_of_link:
                    lane = lane_of_link[index]
                    if lane in signal.lanes:
                        released.add(signal.lanes.index(lane))
            slots.append(sorted(released))
        return slots

    def act(self, observation) -> int:
        signal = self.env.traffic_signals[self.env.ts_ids[0]]
        if self._phase_lane_slots is None:
            self._phase_lane_slots = self._build_phase_map(signal)

        queues = signal.get_lanes_queue()  # normalised queue per incoming lane
        # Serve whichever phase currently has the most vehicles waiting on it.
        scores = [sum(queues[i] for i in slots) for slots in self._phase_lane_slots]
        return int(max(range(len(scores)), key=scores.__getitem__))


class SafetyConstrainedController(Controller):
    """
    Wraps another controller and enforces two constraints it may not respect.

    WHY THIS EXISTS
    ---------------
    REPORT.md lists "nothing here is safety-validated" as a limitation: the
    10 s minimum green and the 4 s amber were the only constraints in force,
    and neither prevents a policy from holding one phase indefinitely or from
    ping-ponging between two phases while a third never runs. A real signal is
    not permitted to do either. Both failure modes are invisible in the average
    waiting time, because a starved approach carries few enough vehicles that
    its misery is diluted by the ones being served well.

    The two constraints:

    MAX_GREEN
        No phase may hold green for longer than this. Caps the worst-case wait
        for anyone facing a red light.

    MAX_STARVATION
        No phase may go unserved for longer than this. Stronger than a maximum
        green, which on its own still permits two phases to alternate forever
        while a third waits. When the bound is hit, the longest-starved phase
        is served next regardless of what the wrapped controller asked for.

    THE EXACT GUARANTEE
    -------------------
    A phase is served within `MAX_STARVATION + (n_phases - 1) * DELTA_TIME`,
    not within MAX_STARVATION exactly. The slack is unavoidable rather than
    sloppy: only one phase can hold green at a time, so if several cross the
    bound on the same step they have to be worked through one decision at a
    time. With four phases and a 5 s interval that is 15 s of slack. The
    wrapper serves the longest-waiting phase first, so the queue of starved
    phases drains in the order that keeps the worst case smallest.

    Measured, not asserted. On the trace seed, the longest a phase went
    unserved:

        greedy-queue  balanced   245 s unwrapped ->  215 s wrapped
        greedy-queue  ns_heavy   320 s unwrapped ->  215 s wrapped
        DQN           balanced   280 s unwrapped ->  210 s wrapped
        DQN           ns_heavy   315 s unwrapped ->  210 s wrapped

    which is the 215 s bound above, reached from well outside it.

    IT MUST WATCH THE SIGNAL, NOT ITS OWN REQUESTS
    ----------------------------------------------
    The clocks are advanced in `_observe` from the phase the signal is
    actually showing. The first version of this class advanced them from the
    phase the inner controller asked for, which is not the same thing - sumo-rl
    refuses a phase change until the current green has run for
    `yellow + min_green` - and the constraint consequently never fired at all.
    See `_observe` for the measurement.

    WHY A WRAPPER RATHER THAN A CHANGE TO THE ENVIRONMENT
    -----------------------------------------------------
    Three reasons. The trained policies do not have to be retrained to gain
    the constraint. The same policy can be measured with and without it, so
    the cost of the safety layer is a number rather than an assertion. And the
    constraint applies identically to every controller that drives its own
    signal, so adding it does not quietly change what is being compared.

    THE ONE CONTROLLER IT CANNOT WRAP
    ---------------------------------
    A fixed-time plan is run by SUMO's own program (`fixed_ts=True`), so this
    class's `act` is never consulted and sumo-rl does not maintain the phase
    index it would need to watch. The wrapper is therefore inert on those
    plans, and says so rather than pretending: `act` passes the action through
    and the counters stay at zero. The plans do satisfy both constraints - a
    104 s cycle serves every phase every 104 s - but that is a property of the
    plan, established from its recorded light strings, not something this
    wrapper measured.
    """

    # A through phase already gets 32 s in the untuned plan, and Webster's
    # search may allocate up to 120 s of cycle; 60 s is comfortably above any
    # green a sensible plan would ask for, so the cap binds only on a policy
    # that is genuinely refusing to let go.
    MAX_GREEN = 60

    # Two full cycles of the untuned 104 s plan. Any approach waiting longer
    # than this is being starved by any reasonable standard.
    MAX_STARVATION = 200

    def __init__(self, inner: Controller, env=None, max_green: int = None,
                 max_starvation: int = None):
        super().__init__(env)
        self.inner = inner
        self.max_green = self.MAX_GREEN if max_green is None else max_green
        self.max_starvation = (self.MAX_STARVATION if max_starvation is None
                               else max_starvation)
        self.name = inner.name + "+safe"
        self.uses_env_control = inner.uses_env_control
        # Counters, in simulated seconds, of what the SIGNAL did.
        self._green_held = 0
        self._since_served = {}
        self._current = None
        self._observed_once = False
        # Diagnostics: how often each constraint had to intervene.
        self.max_green_overrides = 0
        self.starvation_overrides = 0

    def on_episode_start(self, env) -> None:
        self.env = env
        self.inner.on_episode_start(env)
        signal = env.traffic_signals[env.ts_ids[0]]
        # `num_green_phases`, not `len(green_phases)`: sumo-rl only builds the
        # `green_phases` list when it is driving the signal itself. Wrapping a
        # fixed-time plan - where SUMO runs the program and `fixed_ts=True` -
        # leaves the attribute absent, and reading it raised AttributeError
        # before the episode had taken a single step. The count is published in
        # both modes and is the thing actually wanted here.
        self._since_served = {i: 0 for i in range(signal.num_green_phases)}
        self._green_held = 0
        self._current = None
        self._observed_once = False

    def _observe(self) -> None:
        """
        Advance the clocks from what the signal is DOING.

        This is the whole correctness of the wrapper, and the first version got
        it wrong. It reset a phase's clock when the inner controller *asked*
        for that phase. But asking is not being served: sumo-rl refuses a phase
        change until the current green has run for `yellow + min_green`, so a
        controller can request a phase every few seconds and never receive it.
        Measured over one `balanced` hour, the phase the greedy heuristic asked
        for was not the phase then showing green on 573 of 720 decisions.

        The consequence was not a wrong number, it was no constraint at all.
        The wrapper's own worst "unserved" clock reached 175 s, stayed under
        its 200 s bound, and fired zero overrides - while the junction left a
        phase unserved for 245 s. A safety wrapper that reports compliance by
        measuring its own intentions is worse than none, because it is
        believed.

        `green_phase` is the phase index actually in force, and it is checked
        against the recorded light strings in the tests. Amber is not a
        service: during it no phase has right of way, so it neither extends the
        current green nor resets anything.
        """
        signal = self.env.traffic_signals[self.env.ts_ids[0]]
        if self._observed_once:
            # No time has passed before the first decision of an episode.
            for phase in self._since_served:
                self._since_served[phase] += DELTA_TIME
        self._observed_once = True

        if getattr(signal, "is_yellow", False):
            self._green_held = 0
            self._current = None
            return

        current = int(signal.green_phase)
        self._since_served[current] = 0
        if current == self._current:
            self._green_held += DELTA_TIME
        else:
            self._green_held = DELTA_TIME
            self._current = current

    def act(self, observation) -> int:
        if not self.uses_env_control:
            # SUMO is running its own fixed program. Our `act` is never
            # consulted, and sumo-rl does not maintain `green_phase` in that
            # mode, so the wrapper can neither observe the signal nor change
            # it. Pass the action through and leave the counters at zero,
            # rather than keeping a clock that would look like evidence and
            # mean nothing. Addendum E reports this row as an inert wrapper.
            return self.inner.act(observation)

        self._observe()
        requested = self.inner.act(observation)

        # Starvation first: it is the stronger guarantee, and a phase that has
        # waited too long should be served even if the current one has not yet
        # reached its maximum green.
        starved = [p for p, waited in self._since_served.items()
                   if waited >= self.max_starvation and p != requested]
        if starved:
            chosen = max(starved, key=lambda p: self._since_served[p])
            self.starvation_overrides += 1
        elif requested == self._current and self._green_held >= self.max_green:
            # Held too long. Hand over to whichever other phase has waited
            # longest, which is the choice that most reduces the worst wait.
            others = [p for p in self._since_served if p != self._current]
            chosen = max(others, key=lambda p: self._since_served[p])
            self.max_green_overrides += 1
        else:
            chosen = requested

        # Deliberately no clock update here. The clocks move in `_observe`,
        # from what the signal did, and the request made now may not be
        # granted - which is exactly the mistake the first version made.
        return chosen


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------
def route_file(scenario: str, seed: int) -> Path:
    path = ROUTES_DIR / ("%s_seed%d.rou.xml" % (scenario, seed))
    if not path.exists():
        raise FileNotFoundError(
            "%s missing - run:  python networks/generate_routes.py" % path.name
        )
    return path


def make_env(scenario: str, seed: int, fixed_ts: bool, tripinfo: Path | None = None,
             route_path: Path | None = None):
    """
    Build a sumo-rl environment for one episode.

    `fixed_ts=True` tells sumo-rl to leave the signal alone so SUMO runs the
    static plan from the network file - that is how the baseline works.
    """
    from sumo_rl import SumoEnvironment

    if not NET_FILE.exists():
        raise FileNotFoundError(
            "networks/single_intersection.net.xml missing - run:  "
            "python networks/build_network.py"
        )

    # Keep SUMO's own progress chatter off stdout so evaluation output stays
    # readable.
    extra = "--no-step-log true --duration-log.disable true"
    if tripinfo is not None:
        tripinfo.parent.mkdir(parents=True, exist_ok=True)
        # sumo-rl passes this string to SUMO by splitting it on spaces, so the
        # path must not contain any. We therefore hand SUMO a path relative to
        # the repository root and make sure that is the working directory
        # (see the os.chdir in run_episode). The repo may well live under a
        # folder with spaces in its name, which is why this care is needed.
        rel = tripinfo.relative_to(ROOT).as_posix()
        extra += (" --tripinfo-output %s --tripinfo-output.write-unfinished true"
                  % rel)

    return SumoEnvironment(
        net_file=str(NET_FILE),
        route_file=str(route_path or route_file(scenario, seed)),
        use_gui=False,
        num_seconds=EPISODE_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        single_agent=True,
        reward_fn=REWARD_FN,
        fixed_ts=fixed_ts,
        # SUMO has its own randomness (driver imperfection, lane choice).
        # Pinning it to the demand seed makes an episode fully reproducible.
        sumo_seed=seed,
        sumo_warnings=False,
        additional_sumo_cmd=extra,
        add_system_info=True,
        add_per_agent_info=True,
    )


# ---------------------------------------------------------------------------
# Trip-report parsing
# ---------------------------------------------------------------------------
def parse_tripinfo(path: Path) -> dict:
    """
    Read SUMO's per-vehicle trip report.

    Each <tripinfo> element is one vehicle. The fields we use:
        waitingTime - seconds spent with speed below 0.1 m/s (i.e. queued)
        timeLoss    - seconds lost relative to free-flow travel
        duration    - total seconds in the network
        arrival     - -1 if the vehicle was still driving when the hour ended
    """
    if not path.exists():
        return {"vehicles_entered": 0}

    waits, losses, durations, speeds = [], [], [], []
    completed = 0
    for _, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag != "tripinfo":
            continue
        waits.append(float(elem.get("waitingTime", 0.0)))
        losses.append(float(elem.get("timeLoss", 0.0)))
        duration = float(elem.get("duration", 0.0))
        durations.append(duration)
        route_len = float(elem.get("routeLength", 0.0))
        if duration > 0:
            speeds.append(route_len / duration)
        if float(elem.get("arrival", -1)) >= 0:
            completed += 1
        elem.clear()

    n = len(waits)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    return {
        "vehicles_entered": n,
        "throughput_veh": completed,
        "mean_waiting_time_s": mean(waits),
        "mean_time_loss_s": mean(losses),
        "mean_trip_duration_s": mean(durations),
        "mean_speed_mps": mean(speeds),
    }


# ---------------------------------------------------------------------------
# The episode loop
# ---------------------------------------------------------------------------
def run_episode(controller_factory, controller_name: str, scenario: str, seed: int,
                keep_trace: bool = True, quiet: bool = True) -> EpisodeResult:
    """
    Simulate one hour under one controller and return its metrics.

    `controller_factory` is called with the constructed environment and must
    return a controller object exposing `.act(obs)` and `.uses_env_control`.
    Passing a factory rather than a controller lets traffic-responsive
    controllers (which need to read the environment) be built the same way as
    stateless ones.
    """
    tripinfo = TRIPINFO_DIR / ("%s_%s_seed%d.xml" % (
        controller_name.replace("-", "_"), scenario, seed))

    # The environment must know at CONSTRUCTION time whether sumo-rl or SUMO
    # drives the signal, but only the controller knows that. So build one
    # against a placeholder purely to read the flag, then discard it.
    fixed_ts = not controller_factory(None).uses_env_control

    previous_cwd = os.getcwd()
    os.chdir(ROOT)  # so the relative --tripinfo-output path resolves correctly
    try:
        with quiet_stdout():
            env = make_env(scenario, seed, fixed_ts=fixed_ts, tripinfo=tripinfo)
        controller = controller_factory(env)

        result = EpisodeResult(controller=controller_name, scenario=scenario, seed=seed)
        queue_samples = []
        last_green = None

        with quiet_stdout():
            observation, _ = env.reset()
        # Only now does a live SUMO connection exist, so this is the earliest
        # point a controller can inspect or reconfigure the signal.
        controller.on_episode_start(env)
        done = False
        while not done:
            action = controller.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            result.total_reward += float(reward)

            stopped = float(info.get("system_total_stopped", 0.0))
            queue_samples.append(stopped)

            # Count green-to-green changes, ignoring the amber in between, so
            # the number means "how often the right of way was handed over".
            state = _signal_state(env)
            if "y" not in state and state:
                if last_green is not None and state != last_green:
                    result.phase_switches += 1
                last_green = state

            if keep_trace:
                result.trace.append({
                    "sim_time": float(info.get("step", env.sim_step)),
                    "queue_veh": stopped,
                    "system_mean_waiting_time_s":
                        float(info.get("system_mean_waiting_time", 0.0)),
                    "system_mean_speed_mps":
                        float(info.get("system_mean_speed", 0.0)),
                    "reward": float(reward),
                    "signal_state": state,
                })

        # Ask SUMO how many vehicles were still queued to be inserted, before
        # we tear the connection down. A large number here would mean the
        # approaches were so full that demand was being dropped outside the
        # simulation, which would invalidate the comparison.
        #
        # This must not fail to zero. Zero is the *good* reading - it is what
        # the report cites to argue no demand was hidden outside the
        # simulation - so a failed measurement that silently produced 0 would
        # be indistinguishable from a clean result and would corroborate a
        # claim nothing had actually checked. Store the sentinel -1 instead, so
        # a broken reading is visible in the CSV and in any table built from it.
        try:
            result.vehicles_not_inserted = len(
                env.sumo.simulation.getPendingVehicles())
        except traci.TraCIException as exc:
            print("  warning: could not read the insertion backlog for "
                  "%s/%s seed %d (%s); recording -1"
                  % (controller_name, scenario, seed, exc))
            result.vehicles_not_inserted = -1
        env.close()  # flushes the tripinfo file to disk
    finally:
        os.chdir(previous_cwd)

    result.mean_queue_length_veh = (
        sum(queue_samples) / len(queue_samples) if queue_samples else 0.0)
    result.max_queue_length_veh = max(queue_samples) if queue_samples else 0.0

    for key, value in parse_tripinfo(tripinfo).items():
        setattr(result, key, value)

    if not quiet:
        print("  %-13s %-9s seed %-3d  wait %6.1fs  queue %5.1f  through %4d"
              % (controller_name, scenario, seed, result.mean_waiting_time_s,
                 result.mean_queue_length_veh, result.throughput_veh))
    return result


def _signal_state(env) -> str:
    """
    The light string currently displayed, e.g. "GGrrrrGGrrrr".

    We compare these strings rather than a phase *index* because the two
    controllers set the signal by different routes: the fixed-time plan steps
    SUMO's own phase counter, while sumo-rl overwrites the light string
    directly and never touches that counter. Reading the lights themselves is
    the one measurement that means the same thing for both.
    """
    with contextlib.suppress(Exception):
        return env.sumo.trafficlight.getRedYellowGreenState(env.ts_ids[0])
    return ""


@contextlib.contextmanager
def quiet_stdout():
    """
    Silence output written straight to file descriptor 1.

    SUMO is a separate process that inherits our stdout, and it announces
    itself whenever a connection opens. Over a full evaluation sweep that is
    hundreds of lines of noise between the numbers we actually want to read.
    `contextlib.redirect_stdout` cannot help here because it only rebinds
    Python's `sys.stdout`; the child process writes to the underlying file
    descriptor, so the descriptor itself has to be redirected.
    """
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


# ---------------------------------------------------------------------------
# Convenience factories used by the baseline / evaluation scripts
# ---------------------------------------------------------------------------
# Each factory takes the environment (or None, when run_episode is only
# probing for the `uses_env_control` flag) and returns a ready controller.
def fixed_time_factory():
    return lambda env: FixedTimeController(env)


def policy_factory(model, name: str):
    return lambda env: PolicyController(model, name, env)


def greedy_factory():
    return lambda env: GreedyQueueController(env)


def safety_wrapped_factory(inner_factory, max_green: int = None,
                           max_starvation: int = None):
    """
    Wrap any controller factory in the max-green / anti-starvation constraints.

    Takes a factory rather than a controller for the same reason `run_episode`
    does: the wrapped controller may need the environment, which does not exist
    until the episode starts.
    """
    return lambda env: SafetyConstrainedController(
        inner_factory(env), env, max_green, max_starvation)
