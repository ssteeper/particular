"""particular.simulate for many setups at once on an NVIDIA GPU, in float64 (see research/gpu-notes.md).

One CUDA block runs one setup: its positions and velocities sit in shared memory, each thread sums
the forces on its particles over all the others, then the block steps and logs. Setups of any size
and run length share a launch, so nothing is padded or bucketed. The physics is simulate's; only
the rounding differs (rsqrt, fused multiply-adds), and that drifts apart chaotically over a long
run the way any change in summation order does.

Needs torch built with CUDA. torch is imported on first use, so nothing else in the repo needs it.
The kernel is compiled at first use with NVRTC, which ships with torch; no CUDA toolkit is needed.
Each calling thread gets its own CUDA stream, so calls from several threads run on the GPU at once.
"""
import threading

import numpy as np

import particular as P

BATCH = 512             # setups per launch group; the GPU wants hundreds of blocks in flight
PAIRS_PER_LAUNCH = 1e9  # pair-steps per launch (~0.1 s), well inside Windows' 2 s driver watchdog

SOURCE = r"""
extern "C" __global__ void advance(double* state, const int* meta, const long long* offset,
        const int* species, const double* force, double* log,
        int nmax, int smax, int step0, int nsteps, int log_every, double dt)
{
    // state (B, 4, nmax): x, y, vx, vy. meta (B, 2): n, steps. species (B, nmax).
    // force (B, 2, smax, smax): -g * pull and c * push. log: setup b's frames from offset[b].
    extern __shared__ double sh[];
    const int b = blockIdx.x, n = meta[2 * b], steps = meta[2 * b + 1];
    if (step0 >= steps) return;
    const int last = min(step0 + nsteps, steps), ss = smax * smax;
    double *x = sh, *y = x + nmax, *vx = y + nmax, *vy = vx + nmax, *tab = vy + nmax;
    int* sp = (int*)(tab + 2 * ss);
    double* st = state + (size_t)b * 4 * nmax;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] = st[i]; y[i] = st[nmax + i]; vx[i] = st[2 * nmax + i]; vy[i] = st[3 * nmax + i];
        sp[i] = species[(size_t)b * nmax + i];
    }
    for (int k = threadIdx.x; k < 2 * ss; k += blockDim.x) tab[k] = force[(size_t)b * 2 * ss + k];
    __syncthreads();
    double* out = log + offset[b];
    for (int t = step0; t < last; t++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const double xi = x[i], yi = y[i];
            const double* gi = tab + sp[i] * smax;
            const double* ci = gi + ss;
            double ax = 0.0, ay = 0.0;
            for (int j = 0; j < n; j++) {      // same order as numpy's sum over j
                if (j == i) continue;
                const double dx = xi - x[j], dy = yi - y[j];
                const double inv = rsqrt(dx * dx + dy * dy);
                const double coef = inv * inv * inv * (gi[sp[j]] + ci[sp[j]] * inv);
                ax += coef * dx;
                ay += coef * dy;
            }
            vx[i] += ax * dt;
            vy[i] += ay * dt;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            if (t % log_every == 0) {
                double* o = out + ((size_t)(t / log_every) * n + i) * 2;
                o[0] = x[i]; o[1] = y[i];
            }
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        st[i] = x[i]; st[nmax + i] = y[i]; st[2 * nmax + i] = vx[i]; st[3 * nmax + i] = vy[i];
    }
}
"""
_kernels = {}
_compiling = threading.Lock()
_local = threading.local()


def _kernel(torch, device):
    """The compiled kernel for device, built once per device."""
    with _compiling:
        if device.index not in _kernels:
            _kernels[device.index] = _compile(torch, device)
    return _kernels[device.index]


def _compile(torch, device):
    """The kernel, compiled by NVRTC for device's architecture and loaded."""
    import ctypes
    from torch.cuda._utils import _cuda_load_module, _get_gpu_rtc_library
    rtc = _get_gpu_rtc_library()
    major, minor = torch.cuda.get_device_capability(device)
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
    prog = ctypes.c_void_p()
    rtc.nvrtcCreateProgram(ctypes.byref(prog), SOURCE.encode(), b"advance.cu", 0, None, None)
    try:
        if rtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)):
            size = ctypes.c_size_t()
            rtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            msg = ctypes.create_string_buffer(size.value)
            rtc.nvrtcGetProgramLog(prog, msg)
            raise RuntimeError("gpu_sim kernel failed to compile:\n" + msg.value.decode())
        size = ctypes.c_size_t()
        rtc.nvrtcGetCUBINSize(prog, ctypes.byref(size))
        cubin = ctypes.create_string_buffer(size.value)
        rtc.nvrtcGetCUBIN(prog, cubin)
    finally:
        rtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return _cuda_load_module(cubin.raw, ["advance"])["advance"]


def _stream(torch, device):
    """This thread's stream on device."""
    streams = _local.__dict__.setdefault("streams", {})
    if device.index not in streams:
        streams[device.index] = torch.cuda.Stream(device)
    return streams[device.index]


def _forces(s):
    """Species (n,) and the -g * pull, c * push tables (S, S) of a setup, as simulate builds them."""
    n = len(s["pos"])
    species = np.zeros(n, int) if s.get("species") is None else np.asarray(s["species"], int)
    rules = s.get("rules") or P.RULES["classic"]
    return (species, -s.get("g", P.G) * np.asarray(rules["pull"], float),
            s.get("c", P.C) * np.asarray(rules["push"], float))


def _run(torch, kernel, device, setups, log_every):
    B = len(setups)
    ns = [len(s["pos"]) for s in setups]
    nmax = max(ns)
    forces = [_forces(s) for s in setups]
    smax = max(len(f[1]) for f in forces)
    state = np.zeros((B, 4, nmax))
    meta = np.zeros((B, 2), np.int32)
    species = np.zeros((B, nmax), np.int32)
    tables = np.zeros((B, 2, smax, smax))
    sizes = np.zeros(B, np.int64)
    for b, (s, n, (sp, gp, cp)) in enumerate(zip(setups, ns, forces)):
        state[b, :2, :n] = np.asarray(s["pos"], float).T
        state[b, 2:, :n] = np.asarray(s["vel"], float).T
        species[b, :n] = sp
        tables[b, 0, :len(gp), :len(gp)] = gp
        tables[b, 1, :len(cp), :len(cp)] = cp
        meta[b] = n, s["steps"]
        sizes[b] = -(-s["steps"] // log_every) * n * 2
    offset = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    shared = (4 * nmax + 2 * smax * smax) * 8 + nmax * 4
    if shared > 48 * 1024:
        raise ValueError(f"{nmax} particles don't fit in one block's shared memory")

    gpu = [torch.from_numpy(a).to(device) for a in (state, meta, offset, species, tables)]
    log = torch.empty(int(sizes.sum()), dtype=torch.float64, device=device)
    threads = min(1024, -(-nmax // 32) * 32)              # one thread per particle, or a loop
    chunk = max(1, int(PAIRS_PER_LAUNCH // sum(n * n for n in ns)))
    for step0 in range(0, int(meta[:, 1].max()), chunk):
        kernel(grid=(B, 1, 1), block=(threads, 1, 1), shared_mem=shared,
               args=[*gpu, log, nmax, smax, step0, chunk, log_every, float(P.DT)])
    flat = log.cpu().numpy()
    return [flat[o:o + z].reshape(-1, n, 2).copy() for o, z, n in zip(offset, sizes, ns)]


def simulate_batch(setups, log_every=10, device="cuda", batch=BATCH):
    """Run particular.simulate on each setup; returns a list of (frames, n, 2) float64 arrays.

    Each setup is a dict with pos (n, 2), vel (n, 2) and steps, plus optionally g, c, species and
    rules as particular.build returns them (missing ones take simulate's defaults). A setup that
    blows up comes back with non-finite values, as from simulate. batch is the number of setups
    per launch; results don't depend on it, they're bitwise the same however setups are grouped.
    """
    import torch
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    out = [None] * len(setups)
    # largest first: launch groups hold similar sizes, and the longest blocks start first
    order = sorted(range(len(setups)), key=lambda k: -len(setups[k]["pos"]))
    with torch.cuda.device(device), torch.cuda.stream(_stream(torch, device)):
        kernel = _kernel(torch, device)
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            for k, data in zip(idx, _run(torch, kernel, device, [setups[k] for k in idx], log_every)):
                out[k] = data
    return out
