"""Contiguous D2H head groups with double-buffered CPU projection tiles."""
import torch


class HeadReadouts:
    def __init__(self, rows, head_widths, dtype, tile_rows):
        self.capacity = (rows + 1023) // 1024 * 1024
        self.widths = tuple(head_widths)
        self.dtype = dtype
        self.tile_rows = tile_rows
        self.groups = [torch.empty((self.capacity, width), dtype=dtype, pin_memory=True) for width in head_widths]
        self.tiles = [torch.empty((tile_rows, sum(head_widths)), dtype=dtype, pin_memory=True) for _ in range(2)]
        self.events = [None, None]

    def supports(self, rows, head_widths, dtype, tile_rows):
        return rows <= self.capacity and tuple(head_widths) == self.widths and dtype == self.dtype and tile_rows == self.tile_rows

    def store(self, group, values):
        if values.shape[1] != self.widths[group] or not values.is_contiguous():
            raise ValueError('Head readouts must be contiguous [rows, head channels]')
        self.groups[group][:len(values)].copy_(values, non_blocking=True)

    def project(self, module, output, rows):
        # The caller synchronizes once after all D2H groups have been written.
        stream = torch.cuda.current_stream(output.device)
        for index, start in enumerate(range(0, rows, self.tile_rows)):
            slot = index % 2
            if self.events[slot] is not None:
                self.events[slot].synchronize()
            count = min(self.tile_rows, rows - start)
            tile = self.tiles[slot][:count]
            offset = 0
            for group, width in zip(self.groups, self.widths):
                tile[:, offset:offset + width].copy_(group[start:start + count])
                offset += width
            device_values = tile.to(output.device, non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)
            self.events[slot] = done
            yield slice(start, start + count), module(device_values)
