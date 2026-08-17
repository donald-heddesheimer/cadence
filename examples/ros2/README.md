# cadence in a ROS 2 node

A timer-driven node whose callback runs a three-stage CUDA pipeline. It publishes
end-to-end latency on `~/latency_ms` and uses cadence scopes to attribute that
latency to individual stages.

The deadline matches the timer period, so the report shows whether the callback
maintains its requested rate and identifies the stages behind missed deadlines.

This package remains outside the main cadence build and CI so ROS 2 is not a
dependency of the header-only library.

## Build

```sh
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
ln -s /path/to/cadence/examples/ros2 cadence_ros2_example
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
```

Needs `nvcc` on `PATH` (`export PATH=/usr/local/cuda/bin:$PATH`). cadence itself
needs nothing installed: the package prefers `find_package(cadence)` and falls
back to the headers in the tree it ships inside.

## Run

```sh
source install/setup.bash
ros2 run cadence_ros2_example perception_node
ros2 topic echo /cadence_perception/latency_ms      # in another terminal
```

Ctrl-C prints the report. Parameters: `period_ms` (default 5.0), `num_elements`
(default 1048576), `num_taps` (default 384).

## What it shows

Eight seconds at the defaults on an RTX A4000:

```
  label      scope      n    mean     p50     p95     max  jitter  distribution
  ─────────────────────────────────────────────────────────────────────────────
  callback   host    1572  1.95ms  1.99ms  2.01ms  7.80ms  6.05ms  █          ▂
  normalize  device  1572  31.4µs  27.6µs  28.7µs  5.98ms  5.95ms  █          ▂
  normalize  host    1572  13.2µs  9.06µs  10.3µs  5.99ms  5.98ms  █          ▂
  detect     device  1572  1.87ms  1.91ms  1.92ms  1.93ms   252µs  ▃       ▂▂▃█
  detect     host    1572  6.17µs  6.10µs  6.85µs  22.2µs  17.2µs  █▃▂▂       ▂
  threshold  device  1572  25.6µs  25.6µs  26.6µs  26.9µs  2.34µs  ▃▂   █▂▂  ▃▂
  threshold  host    1572  6.18µs  6.13µs  6.91µs  10.2µs  5.34µs  ▂▄█▆▃▂▂▂▂▂ ▂

  deadline  5.00ms on callback (host)
  MISSED    1571/1572 iterations inside budget (99.9%); worst 7.80ms at 156% of budget
            [████████████████░░░░░░░░░░░░░░░░░░░░░░░░]  p95 2.01ms at 40%

  slowest iterations
    #270    7.80ms  normalize 28.7µs · detect 1.68ms · threshold 25.6µs
    #269    5.99ms  normalize 5.98ms · detect 1.69ms · threshold 25.6µs
```

The p95 is 2.01ms against a 5ms deadline, and 1571 of 1572 callbacks complete
inside budget. The mean latency topic alone would report 1.95ms.

The stage breakdown adds the missing context. Iteration #269 spends 5.98ms in
`normalize`, whose median is 27.6µs. Iteration #270 takes 7.80ms while each GPU
stage remains near its usual duration, indicating delay outside GPU execution.

## Taking the rate away from it

Reduce the period until the workload no longer fits:

```sh
ros2 run cadence_ros2_example perception_node --ros-args -p period_ms:=2.0
```

```
  deadline  2.00ms on callback (host)
  MISSED    3621/3938 iterations inside budget (92.0%); worst 2.40ms at 120% of budget
            [████████████████████████████████████████]  p95 2.00ms at 100%
```

Here 92% of callbacks complete inside budget and p95 reaches the deadline. The
1.98ms mean remains below the 2.00ms budget, demonstrating why the report counts
individual misses.

## Notes

- `CADENCE_FLUSH()` is called at the `cudaStreamSynchronize` the callback was
  going to do anyway. That is the point of the deferred design: the records
  resolve against work that has already finished, so nothing is waited on twice.
- The budget names no label. The only label that records host time and launches
  nothing is the `CADENCE_SCOPE` around the callback, and cadence resolves the
  deadline to it on its own.
- A trace of the slowest iterations is written to `cadence_perception.json` in
  the working directory. Open it at [ui.perfetto.dev](https://ui.perfetto.dev).
- If the report warns about dropped host spans, the machine's monotonic clock did
  not advance across them. It happens on virtualized hosts and says something
  about the machine rather than about the node; the GPU figures are unaffected.
