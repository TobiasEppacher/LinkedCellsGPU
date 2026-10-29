#!/bin/bash
# Counts the operations that the GPU kernels of every container rebuild really
# execute, using the Nsight Compute profiler (ncu). Two rebuilds are profiled per
# implementation: the first sorts the random initial order, the second finds the
# particles already sorted.
#
# Needs an NVIDIA GPU, Nsight Compute's ncu on PATH and the binaries listed in
# kernels.txt, built in the build directories named there. From the project root:
#                GPU=<idle gpu> bash scripts/ncu-rebuild/ncu-rebuild.sh
#
# Options (environment variables):
#   GPU=1        which GPU to use (pick an idle one, see nvidia-smi)
#   CELLS=16     interior cells per dimension (domain edge = CELLS * interaction_radius)
#   PARTICLES=125000  number of particles, placed at random (fixed seed)
#                CELLS and PARTICLES take lists ("16 32" or 16,32): every combination
#                is run into <out>/c<cells>_p<particles>/
#   OUT=<dir>    where the results go (default: scripts/ncu-rebuild/out)
#
# Result: <out>/ncu/     one full ncu report per implementation
#         <out>/raw/     the console output of each run
#         <out>/configs/ the workload each implementation ran with
#
# The two rebuilds appear in launch order in each report, each starting with its
# cell index kernel (updateCellIdx):
#   1. unsorted: the random initial order -- a full permutation, scattered accesses
#   2. sorted:   particles already in cell order (only_forces never moves them) --
#                the identity permutation, the steady-state best case
#
# Notes:
#   - The scan takes a different number of launches per backend: cub runs an init
#     and a scan kernel, Kokkos' parallel_scan two passes, the SYCL scan two kernels
#     per recursion level plus the top level. SYCL also clears the counts with a
#     fill kernel.
#   - Memsets and device-to-device copies (zeroing the counts in CUDA/Kokkos,
#     seeding the scatter cursor) are not kernels, so ncu does not record them.
#   - Durations in the reports are measured under the profiler, which replays each
#     kernel several times. They are not benchmark numbers.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$HERE/out}"
GPU="${GPU:-0}"
CELLS="${CELLS:-16}"
PARTICLES="${PARTICLES:-125000}"

CELLS_LIST=$(echo "$CELLS" | tr ',' ' ')
PARTICLES_LIST=$(echo "$PARTICLES" | tr ',' ' ')
for var in CELLS_LIST PARTICLES_LIST; do
    if [ -z "$(echo ${!var})" ]; then
        echo "${var%_LIST} must not be empty" >&2
        exit 1
    fi
    for value in ${!var}; do
        if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            echo "${var%_LIST} takes positive integers, got '$value'" >&2
            exit 1
        fi
    done
done

# More than one combination: one run of this script per combination, each in its own directory.
if [ "$(echo $CELLS_LIST $PARTICLES_LIST | wc -w)" -gt 2 ]; then
    for c in $CELLS_LIST; do
        for p in $PARTICLES_LIST; do
            echo "=== c${c}_p${p}"
            CELLS=$c PARTICLES=$p OUT="$OUT/c${c}_p${p}" bash "$0" || exit 1
        done
    done
    exit 0
fi
CELLS=$(echo $CELLS_LIST)
PARTICLES=$(echo $PARTICLES_LIST)

# workload.yaml fixes box_min; box_max is the domain inset by the same margin.
RADIUS=$(awk '$1 == "interaction_radius:" { print $2; exit }' "$HERE/workload.yaml")
MARGIN=$(awk '$1 == "box_min:" { gsub(/[][,]/, "", $2); print $2; exit }' "$HERE/workload.yaml")
DOMAIN=$(awk -v c="$CELLS" -v r="$RADIUS" 'BEGIN { printf "%.10g", c * r }')
BOX_MAX=$(awk -v d="$DOMAIN" -v m="$MARGIN" 'BEGIN { printf "%.10g", d - m }')

command -v ncu >/dev/null || { echo "ncu not found; put Nsight Compute on PATH" >&2; exit 1; }

export OMP_PROC_BIND=spread OMP_PLACES=threads
export CUDA_VISIBLE_DEVICES="$GPU"

# --- what to count -----------------------------------------------------------
# Collected on top of --set full, so every report also holds these counters.
# "thread instructions" are counted per thread and summed over all threads.
METRICS="
float_add            smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
float_mul            smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
float_fma            smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
float32_all          smsp__sass_thread_inst_executed_op_fp32_pred_on.sum
double_add           smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
double_mul           smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
double_fma           smsp__sass_thread_inst_executed_op_dfma_pred_on.sum
float64_all          smsp__sass_thread_inst_executed_op_fp64_pred_on.sum
integer              smsp__sass_thread_inst_executed_op_integer_pred_on.sum
bit_logic            smsp__sass_thread_inst_executed_op_bit_pred_on.sum
conversion           smsp__sass_thread_inst_executed_op_conversion_pred_on.sum
control_flow         smsp__sass_thread_inst_executed_op_control_pred_on.sum
memory               smsp__sass_thread_inst_executed_op_memory_pred_on.sum
misc                 smsp__sass_thread_inst_executed_op_misc_pred_on.sum
warp_shuffle         smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum
thread_inst_total    sm__sass_thread_inst_executed.sum
warp_inst_total      smsp__inst_executed.sum
warp_global_load     smsp__sass_inst_executed_op_global_ld.sum
warp_global_store    smsp__sass_inst_executed_op_global_st.sum
warp_shared_load     smsp__sass_inst_executed_op_shared_ld.sum
warp_shared_store    smsp__sass_inst_executed_op_shared_st.sum
warp_local_load      smsp__sass_inst_executed_op_local_ld.sum
warp_local_store     smsp__sass_inst_executed_op_local_st.sum
warp_atomic          smsp__sass_inst_executed_op_atom.sum
warp_branch          smsp__sass_inst_executed_op_branch.sum
global_ld_requests   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
global_ld_sectors    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
global_st_requests   l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
global_st_sectors    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
dram_bytes_read      dram__bytes_read.sum
dram_bytes_write     dram__bytes_write.sum
thread_blocks        launch__grid_size
duration_ns_profiled gpu__time_duration.sum
threads_per_block    launch__block_size
registers_per_thread launch__registers_per_thread
shared_bytes_static  launch__shared_mem_per_block_static
shared_bytes_dynamic launch__shared_mem_per_block_dynamic
blocks_per_sm_limit  launch__occupancy_limit_registers
waves_per_sm         launch__waves_per_multiprocessor
"

# One comma separated list for the ncu command line.
METRIC_LIST=$(echo "$METRICS" | awk 'NF == 2 { printf "%s%s", sep, $2; sep = "," }')

# --- profile every implementation --------------------------------------------
IMPLEMENTATIONS=$(grep -v '^ *#' "$HERE/kernels.txt" | awk 'NF == 4')

mkdir -p "$OUT/raw" "$OUT/configs" "$OUT/ncu"

echo "workload: ${CELLS}^3 cells, $PARTICLES particles"
echo "$IMPLEMENTATIONS" | while read -r name algorithm binary kernel; do
    echo "profiling $name ..."
    config="$OUT/configs/$name.yaml"
    sed -e "s|^algorithm:.*|algorithm: $algorithm|" \
        -e "s|^domain_size:.*|domain_size: [$DOMAIN, $DOMAIN, $DOMAIN]|" \
        -e "s|^\( *box_max:\).*|\1 [$BOX_MAX, $BOX_MAX, $BOX_MAX]|" \
        -e "s|^\( *num_particles:\).*|\1 $PARTICLES|" \
        "$HERE/workload.yaml" > "$config"

    # --kernel-name-base mangled: every AdaptiveCpp kernel is called
    # "__acpp_kernel", only the mangled name says which container it is.
    # --set full adds all sections for ncu-ui.
    ncu --set full --metrics "$METRIC_LIST" \
        --kernel-name-base mangled -k "regex:$kernel" \
        -o "$OUT/ncu/$name" -f \
        "$ROOT/$binary" "$config" > "$OUT/raw/$name.txt" 2>&1 < /dev/null
done

echo
echo "ncu    : $OUT/ncu/"
echo "raw    : $OUT/raw/"
echo "configs: $OUT/configs/"
