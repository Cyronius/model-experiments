"""Packed-text host for the fused2 encoder: many texts share each 128-token dispatch.

Two input/output buffer sets alternate, so the host prepares the next dispatch and pools
the previous one while the NPU runs (td_npu2's asynchronous tdn_start/tdn_wait)."""
import ctypes as ct
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from encoder import ROOT, digest
from arithmetic import pack_weights
from fused_encoder import parameter_packet
from runtime import load_runtime
from model import Student, tern_codes
from query_head import install_query_head, query_role

BUILD = ROOT/'build_f2_L3_cap128_d2'
LIBRARY = ROOT/'host/out/libtd_npu2.so'
SLOT = 128
SETS = 2
PACKETS = 170


def weight_stream2(model):
    """Per layer, 170 packets of 4096 bytes in consumption order. A weight packet holds one
    32-output-channel block of packed ternary codes, then its bias and scale/4 at byte 3072."""
    packets = []

    def linear(codes, scale, bias):
        assert len(scale) == len(bias) == codes.shape[0]
        for n, block in enumerate(pack_weights(codes)):
            p = np.zeros(4096, np.uint8)
            p[:block.size] = block.reshape(-1)
            f = p[3072:].view(np.float32)
            f[:32] = bias[n*32:(n+1)*32]; f[32:64] = scale[n*32:(n+1)*32] / 4
            packets.append(p)

    for layer in model.layers:
        start = len(packets)
        for name in ('q', 'k', 'v', 'o'):
            m = getattr(layer, name)
            c, s = tern_codes(m.weight)
            linear(c.numpy().astype(np.int8), s.numpy().reshape(-1), m.bias.detach().numpy())
        packets.append(parameter_packet(layer.ln1.weight.detach().numpy(), layer.ln1.bias.detach().numpy()))
        uc, us = tern_codes(layer.fc1.weight); dc, ds = tern_codes(layer.fc2.weight)
        us, ds = us.numpy().reshape(-1), ds.numpy().reshape(-1)
        for g in range(6):
            linear(uc[g*256:(g+1)*256].numpy().astype(np.int8), us[g*256:(g+1)*256],
                   layer.fc1.bias.detach().numpy()[g*256:(g+1)*256])
            bias = layer.fc2.bias.detach().numpy() if g == 0 else np.zeros(384, np.float32)
            linear(dc[:, g*256:(g+1)*256].numpy().astype(np.int8), ds, bias)
        packets.append(parameter_packet(layer.ln2.weight.detach().numpy(), layer.ln2.bias.detach().numpy()))
        assert len(packets) - start == PACKETS
    return np.stack(packets).reshape(-1)


def pack(lengths, align=1):
    """Best-fit decreasing: assign each text to a 128-token slot; returns (slot, offset) per text.

    With align=8 each text starts on an 8-token boundary, so no 8-key block-floating
    group mixes two texts. Measured on the KJV, this did not change accuracy."""
    span = [-(-n // align) * align for n in lengths]
    by_room = {}                      # free tokens -> slots with exactly that much room
    room, place = [], [None] * len(lengths)
    for i in sorted(range(len(lengths)), key=lambda i: -span[i]):
        need = span[i]
        if need > SLOT:
            raise ValueError(f'text {i} has {lengths[i]} tokens; the slot holds {SLOT}')
        slot = next((by_room[c].pop() for c in range(need, SLOT + 1) if by_room.get(c)), None)
        if slot is None:
            slot = len(room); room.append(SLOT)
        place[i] = (slot, SLOT - room[slot])
        room[slot] -= need
        by_room.setdefault(room[slot], []).append(slot)
    return place, len(room)


class Fused2Encoder:
    def __init__(self, checkpoint, build=BUILD):
        self.model = Student.load(checkpoint).eval()
        install_query_head(self.model, checkpoint)
        self.layers = self.model.dims['layers']
        self.manifest = json.loads((build/'build.json').read_text())
        if self.manifest['layers'] != self.layers or self.manifest['cap'] != SLOT or self.manifest.get('packets') != PACKETS:
            raise ValueError('Compiled encoder shape differs from checkpoint')
        for name, sha in self.manifest['source_sha256'].items():
            if digest(ROOT/name) != sha:
                raise ValueError(f'Rebuild changed source {name}')
        for name, sha in self.manifest['artifact_sha256'].items():
            if digest(build/name) != sha:
                raise ValueError(f'Changed build artifact {name}')
        self.dll, _ = load_runtime(LIBRARY)
        self.handle = self.dll.tdn_open(os.fsencode(build/'final.xclbin'), os.fsencode(build/'insts.bin'))
        try:
            weights = weight_stream2(self.model)
            self.buffers = [self.alloc(weights.nbytes)]
            self.buffers[0][:] = weights
            self.dll.tdn_sync(self.handle, 0, 1, self.buffers[0].nbytes, 0)
            # Per set: segment IDs, input states, K/V scratch, layer outputs.
            self.sets = []
            for _ in range(SETS):
                idx = []
                for size in (4096, SLOT*384*4, 32*8192, self.layers*32*8192):
                    idx.append(len(self.buffers)); self.buffers.append(self.alloc(size))
                run = self.dll.tdn_bind(self.handle, 5, (ct.c_int*5)(0, *idx))
                self.sets.append((run, idx))
        except BaseException:
            self.close()
            raise
        self.dispatch_ms = []

    def alloc(self, size):
        ptr = self.dll.tdn_buf(self.handle, size)
        return np.ctypeslib.as_array((ct.c_uint8*size).from_address(ptr))

    @torch.inference_mode()
    def start(self, s, ids, pos, seg):
        """Queue one dispatch on buffer set s: token ids, positions and segment IDs (0 = padding)."""
        m = self.model
        run, (mask, x, _, _) = self.sets[s]
        states = m.proj(m.ln_e(m.word(ids) + m.pos.weight[pos] + m.tok_type.weight[0]))
        self.buffers[mask].view(np.int32)[:SLOT] = seg
        self.buffers[x].view(np.float32)[:] = states.numpy().reshape(32, 4, 48, 8).transpose(0, 2, 1, 3).reshape(-1)
        for i in (mask, x):
            self.dll.tdn_sync(self.handle, i, 1, self.buffers[i].nbytes, 0)
        self.dll.tdn_start(self.handle, run)

    def finish(self, s):
        """Wait for set s and return its final-layer token states, shape (128, 384)."""
        run, (_, _, _, out) = self.sets[s]
        ms = self.dll.tdn_wait(self.handle, run)
        if ms <= 0:
            raise RuntimeError(f'Incomplete fused2 dispatch: {ms}')
        self.dispatch_ms.append(ms)
        layer = self.buffers[out].nbytes // self.layers
        self.dll.tdn_sync(self.handle, out, 0, layer, (self.layers - 1) * layer)
        raw = self.buffers[out][(self.layers - 1) * layer:].view(np.float32).reshape(32, 2048)[:, :1536]
        return torch.from_numpy(raw.reshape(32, 48, 4, 8).transpose(0, 2, 1, 3).reshape(SLOT, 384).copy())

    def forward(self, ids, pos, seg):
        self.start(0, ids, pos, seg)
        return self.finish(0)

    @torch.inference_mode()
    def encode(self, token_ids, query=False, align=1):
        """Vectors for pre-tokenized texts (lists of ids including [CLS]/[SEP], at most 128 each)."""
        m = self.model
        place, slots = pack([len(t) for t in token_ids], align)
        members = [[] for _ in range(slots)]
        for i, (s, off) in enumerate(place):
            members[s].append((i, off, len(token_ids[i])))
        pooled = torch.empty((len(token_ids), 384))

        def pool(s, group):
            states = self.finish(s)
            for i, off, n in group:
                pooled[i] = states[off] if m.pool == 'cls' else states[off:off + n].mean(0)

        pending = None
        for k, group in enumerate(members):
            ids = np.zeros(SLOT, np.int64); pos = np.zeros(SLOT, np.int64); seg = np.zeros(SLOT, np.int32)
            for j, (i, off, n) in enumerate(group):
                ids[off:off + n] = token_ids[i]; pos[off:off + n] = np.arange(n); seg[off:off + n] = j + 1
            self.start(k % SETS, torch.from_numpy(ids), torch.from_numpy(pos), seg)
            if pending:
                pool(*pending)
            pending = (k % SETS, group)
        if pending:
            pool(*pending)
        with query_role(m, query):
            vectors = F.normalize(m.out(pooled), dim=-1)
        if not torch.isfinite(vectors).all():
            raise ValueError('Nonfinite fused2 output')
        return vectors, slots

    def close(self):
        if getattr(self, 'handle', None):
            self.dll.tdn_close(self.handle); self.handle = None
