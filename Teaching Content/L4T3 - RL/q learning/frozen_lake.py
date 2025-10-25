# frozenlake_arcade_v3_final.py
import arcade
import gymnasium as gym
import numpy as np
from typing import Tuple

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

COLOR_BG = (18, 18, 24)
COLOR_GRID = (40, 45, 60)
COLOR_SAFE = (80, 180, 255)
COLOR_START = (120, 220, 120)
COLOR_HOLE = (30, 30, 30)
COLOR_GOAL = (255, 215, 0)
COLOR_PLAYER = (255, 100, 100)
COLOR_TEXT = (230, 230, 240)
UI_HEIGHT = 90


class FrozenLakeArcade(arcade.Window):
    def __init__(self, map_name="4x4", is_slippery=False):
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.is_slippery)
        self.nrow, self.ncol = self._grid_shape()

        self.tile_px = 96
        width = self.ncol * self.tile_px
        height = self.nrow * self.tile_px + UI_HEIGHT
        super().__init__(width=width, height=height, title="FrozenLake — Arcade v3")

        arcade.set_background_color(COLOR_BG)

        self.obs = None
        self.done = False
        self.last_reward = 0.0
        self.episode = 1
        self.total_return = 0.0

        self._ui_font_size = 16
        self._title_font_size = 18

        self._reset_env()

    # --- Env helpers ---
    def _grid_shape(self) -> Tuple[int, int]:
        return self.env.unwrapped.desc.shape

    def _decode_cell(self, r, c):
        val = self.env.unwrapped.desc[r, c]
        return val.decode("utf-8") if isinstance(val, (bytes, np.bytes_)) else str(val)

    def _state_to_rc(self, s):
        return s // self.ncol, s % self.ncol

    def _reset_env(self):
        self.obs, _ = self.env.reset()
        self.done = False
        self.last_reward = 0.0
        self.total_return = 0.0

    def _rebuild_env(self):
        try:
            self.env.close()
        except Exception:
            pass
        self.env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.is_slippery)
        self.nrow, self.ncol = self._grid_shape()
        self.tile_px = max(48, min(96, int(768 / max(self.nrow, self.ncol))))
        self.set_size(self.ncol * self.tile_px, self.nrow * self.tile_px + UI_HEIGHT)
        self._reset_env()

    # --- Draw ---
    def on_draw(self):
        self.clear()              # <-- replaces start_render()
        self._draw_grid()
        self._draw_ui()


    def _draw_grid(self):
        pad = 2
        for r in range(self.nrow):
            for c in range(self.ncol):
                cell = self._decode_cell(r, c)
                x0 = c * self.tile_px + pad
                y0 = (self.nrow - 1 - r) * self.tile_px + UI_HEIGHT + pad
                w = self.tile_px - 2 * pad
                h = self.tile_px - 2 * pad

                color = {
                    "S": COLOR_START,
                    "F": COLOR_SAFE,
                    "H": COLOR_HOLE,
                    "G": COLOR_GOAL,
                }.get(cell, COLOR_SAFE)

                rect = arcade.rect.XYWH(x0, y0, w, h)
                arcade.draw_rect_filled(rect, color)
                arcade.draw_rect_outline(rect, COLOR_GRID, 2)

        # Player
        pr, pc = self._state_to_rc(int(self.obs))
        px = pc * self.tile_px + self.tile_px / 2
        py = (self.nrow - 1 - pr) * self.tile_px + UI_HEIGHT + self.tile_px / 2
        r = self.tile_px * 0.25
        arcade.draw_circle_filled(px, py, r, COLOR_PLAYER)
        arcade.draw_circle_outline(px, py, r, COLOR_GRID, 2)

    def _draw_ui(self):
        # Draw footer background (use rect API)
        ui_rect = arcade.rect.XYWH(0, 0, self.width, UI_HEIGHT)
        arcade.draw_rect_filled(ui_rect, (25, 28, 35))
        arcade.draw_line(0, UI_HEIGHT, self.width, UI_HEIGHT, (60, 65, 80), 2)

        status = "Done" if self.done else "Playing"
        slippery = "on" if self.is_slippery else "off"
        text1 = f"FrozenLake {self.map_name} — {status} — slippery: {slippery} — episode: {self.episode}"
        text2 = (
            f"Last reward: {self.last_reward:.0f}   Total return: {self.total_return:.0f}   "
            f"Controls: ←↑→↓ move, R reset, Tab toggle slippery, M change map"
        )
        arcade.draw_text(text1, 10, UI_HEIGHT - 28, COLOR_TEXT, self._title_font_size)
        arcade.draw_text(text2, 10, UI_HEIGHT - 56, COLOR_TEXT, self._ui_font_size)

        if self.done:
            msg = "You reached the GOAL! 🎉" if self.last_reward > 0 else "You fell in a HOLE 💀"
            arcade.draw_text(
                msg + "  Press R to start a new episode.",
                self.width // 2, 16, COLOR_TEXT, self._ui_font_size, anchor_x="center"
            )

    # --- Input ---
    def on_key_press(self, key, modifiers):
        if key in (arcade.key.LEFT, arcade.key.RIGHT, arcade.key.UP, arcade.key.DOWN) and not self.done:
            action = {
                arcade.key.LEFT: LEFT,
                arcade.key.DOWN: DOWN,
                arcade.key.RIGHT: RIGHT,
                arcade.key.UP: UP,
            }[key]
            self._step_env(action)
        elif key == arcade.key.R:
            self.episode += 1 if self.done else 0
            self._reset_env()
        elif key == arcade.key.TAB:
            self.is_slippery = not self.is_slippery
            self._rebuild_env()
        elif key == arcade.key.M:
            self.map_name = "8x8" if self.map_name == "4x4" else "4x4"
            self._rebuild_env()
        elif key == arcade.key.ESCAPE:
            self.close()

    def _step_env(self, action):
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        self.obs = new_obs
        self.last_reward = float(reward)
        self.total_return += float(reward)
        self.done = terminated or truncated

    def on_close(self):
        try:
            self.env.close()
        finally:
            return super().on_close()


def main():
    window = FrozenLakeArcade(map_name="4x4", is_slippery=False)
    arcade.run()


if __name__ == "__main__":
    main()
