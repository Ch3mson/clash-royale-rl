import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ArenaConfig:
    screen_width: int
    screen_height: int
    arena_top: int
    arena_bottom: int
    arena_left: int
    arena_right: int

class GridSystem:
    def __init__(self, config: ArenaConfig, grid_rows: int=32, grid_cols: int=18):
        self.config = config
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_width = (config.arena_right - config.arena_left) / grid_cols
        self.cell_height = (config.arena_bottom - config.arena_top) / grid_rows

    def grid_to_pixel(self, row: int, col: int) -> Tuple[int, int]:
        x = int(self.config.arena_left + (col + 0.5) * self.cell_width)
        y = int(self.config.arena_top + (row + 0.5) * self.cell_height)
        return x, y
    
    def pixel_to_grid(self, x: int, y: int) -> Tuple[int, int]:
        col = int((x - self.config.arena_left) / self.cell_width)
        row = int((y - self.config.arena_top) / self.cell_height)
        return row, col
    
    def get_valid_positions(self) ->list:
        positions = list()
        for r in range(5, self.grid_rows):
            for c in range(self.grid_cols):
                positions.append((r, c))
        return positions