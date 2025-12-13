#!/usr/bin/env python
"""
Brain Block Puzzle Solver with Enhanced Interactive GUI
Allows users to place initial pieces before solving
Modified: Click once to drag, click/space to place, ESC to cancel
"""

import sys
from itertools import islice
import dlx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# PUZZLE CONFIGURATIONS
# ============================================================================

PUZZLE_SETS = {
    1: {
        'board': (3, 2),
        'pieces': [
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (1, 0)],
            [(0, 0)]
        ]
    },
    2: {
        'board': (8, 5),
        'pieces': [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(0, 0), (0, 1), (0, 2), (1, 0)],
            [(0, 0), (0, 1), (0, 2), (1, 0)],
            [(0, 0), (1, 0), (1, 1), (2, 0)],
            [(0, 0), (1, 0), (1, 1), (2, 0)],
            [(0, 0), (1, 0), (1, 1), (2, 1)],
            [(0, 0), (1, 0), (1, 1), (2, 1)],
        ]
    },
    3: {
        'board': (10, 6),
        'pieces': [
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
            [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1)],
            [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
            [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
            [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)],
            [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            [(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)],
            [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)],
            [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)],
        ]
    }
}

PUZZLE_SET = 3

# Enhanced color palette - vibrant and distinct colors
PIECE_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
    '#E63946', '#06FFA5', '#FFB703', '#8338EC', '#3A86FF',
    '#FB5607', '#C77DFF', '#06D6A0', '#FF006E', '#FFBE0B'
]

# ============================================================================
# GEOMETRIC TRANSFORMATIONS
# ============================================================================

def normalize(coords):
    """Translate shape to start at (0,0) and sort coordinates."""
    if not coords:
        return tuple()
    min_x = min(x for x, y in coords)
    min_y = min(y for x, y in coords)
    shifted = [(x - min_x, y - min_y) for x, y in coords]
    return tuple(sorted(shifted))


def translate(shape, dx=0, dy=0):
    """Move shape by (dx, dy)."""
    return tuple((x + dx, y + dy) for x, y in shape)


def rotate_90(coords):
    """Rotate coordinates 90 degrees counterclockwise."""
    return [(-y, x) for x, y in coords]


def mirror_horizontal(coords):
    """Mirror coordinates horizontally (flip left-right)."""
    return [(-x, y) for x, y in coords]


def generate_orientations(coords):
    """Generate all unique orientations (rotations + reflections)."""
    orientations = set()
    
    for flip in [False, True]:
        shape = mirror_horizontal(coords) if flip else list(coords)
        current = shape
        for _ in range(4):
            normalized = normalize(current)
            orientations.add(normalized)
            current = rotate_90(current)
    
    return orientations


def generate_placements(coords, board_width, board_height, piece_index):
    """Generate all valid placements of a piece on the board."""
    placements = {}
    orientations = generate_orientations(coords)
    
    for orientation in orientations:
        max_x = max(x for x, y in orientation)
        max_y = max(y for x, y in orientation)
        
        for dx in range(board_width - max_x):
            for dy in range(board_height - max_y):
                placement = translate(orientation, dx, dy)
                placements[placement] = piece_index
    
    return placements


# ============================================================================
# INTERACTIVE PIECE CLASS
# ============================================================================

class InteractivePiece:
    """Represents a draggable, rotatable puzzle piece."""
    
    def __init__(self, piece_coords, piece_index, color, position=(0, 0)):
        self.original_coords = normalize(piece_coords)
        self.coords = self.original_coords
        self.piece_index = piece_index
        self.color = color
        self.position = position  # (x, y) offset
        self.patches = []
        self.selected = False
        self.placed = False  # Whether piece is on the board
        self.dragging = False  # Whether piece is being dragged
        self.invalid_flash = 0  # Counter for invalid placement flash effect
        self.original_position = None  # Store position before dragging
        self.original_placed = False  # Store placed status before dragging
        
    def rotate(self):
        """Rotate piece 90 degrees."""
        rotated = rotate_90(list(self.coords))
        self.coords = normalize(rotated)
        
    def get_board_coords(self):
        """Get actual board coordinates of the piece."""
        return [(x + self.position[0], y + self.position[1]) 
                for x, y in self.coords]
    
    def contains_point(self, x, y):
        """Check if point is inside this piece."""
        board_coords = self.get_board_coords()
        for px, py in board_coords:
            if abs(x - (px + 0.5)) < 0.5 and abs(y - (py + 0.5)) < 0.5:
                return True
        return False
    
    def move_to(self, x, y):
        """Move piece to new position."""
        # Find offset to place piece near click point
        if self.coords:
            min_x = min(cx for cx, cy in self.coords)
            min_y = min(cy for cx, cy in self.coords)
            self.position = (int(x) - min_x, int(y) - min_y)


# ============================================================================
# ENHANCED INTERACTIVE GUI
# ============================================================================

class PuzzleGUI:
    """Enhanced Interactive GUI for setting up initial puzzle state."""
    
    def __init__(self, board_width, board_height, pieces):
        self.board_width = board_width
        self.board_height = board_height
        self.original_pieces = pieces
        
        # Create interactive pieces with enhanced colors
        self.pieces = []
        palette_y = 0
        for i, piece_coords in enumerate(pieces):
            color = PIECE_COLORS[i % len(PIECE_COLORS)]
            pos = (-6, palette_y)
            self.pieces.append(InteractivePiece(piece_coords, i, color, pos))
            
            max_y = max(y for x, y in normalize(piece_coords))
            palette_y += max_y + 2
        
        self.selected_piece = None
        self.drag_offset = (0, 0)
        self.solver_solutions = []
        self.current_solution_idx = 0
        self.flash_timer = None  # Timer for flash animation
        
        # Setup matplotlib figure with dark theme
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 10), facecolor='#1a1a2e')

        # Create custom grid layout
        gs = self.fig.add_gridspec(1, 3, width_ratios=[0.6, 1.2, 1.2],
                                   left=0.05, right=0.98, wspace=0.15)

        self.ax_palette = self.fig.add_subplot(gs[0])
        self.ax_setup = self.fig.add_subplot(gs[1])
        self.ax_solution = self.fig.add_subplot(gs[2])
        
        self._setup_axes()
        self._setup_buttons()
        self._connect_events()
        self._draw_all()
        
    
    def _setup_axes(self):
        """Configure plot axes with enhanced styling."""
        # Palette area
        self.ax_palette.set_xlim(-8, 2)
        self.ax_palette.set_ylim(-2, 50)
        self.ax_palette.set_aspect('equal')
        self.ax_palette.set_title('PIECE PALETTE', fontsize=16,
                                  fontweight='bold', color='#4ECDC4', pad=15)
        self.ax_palette.set_facecolor('#16213e')
        self.ax_palette.grid(True, alpha=0.2, linestyle='--', color='#4ECDC4')
        self.ax_palette.tick_params(colors='#4ECDC4', labelsize=8)

        # Add decorative border
        for spine in self.ax_palette.spines.values():
            spine.set_edgecolor('#4ECDC4')
            spine.set_linewidth(2)

        # Setup Board area
        self.ax_setup.set_xlim(-0.5, self.board_width + 0.5)
        self.ax_setup.set_ylim(-0.5, self.board_height + 0.5)
        self.ax_setup.set_aspect('equal')
        self.ax_setup.set_title('SETUP BOARD', fontsize=16,
                               fontweight='bold', color='#FFA07A', pad=15)
        self.ax_setup.set_facecolor('#0f3460')

        # Enhanced grid with gradient effect
        for x in range(self.board_width + 1):
            alpha = 0.6 if x % 2 == 0 else 0.3
            self.ax_setup.axvline(x, color='#FFA07A', linewidth=1.5, alpha=alpha)
        for y in range(self.board_height + 1):
            alpha = 0.6 if y % 2 == 0 else 0.3
            self.ax_setup.axhline(y, color='#FFA07A', linewidth=1.5, alpha=alpha)

        # Add corner markers
        marker_size = 0.15
        for x, y in [(0, 0), (0, self.board_height),
                     (self.board_width, 0), (self.board_width, self.board_height)]:
            self.ax_setup.plot(x, y, 'o', color='#FFA07A', markersize=10,
                             markeredgewidth=2, markeredgecolor='white')

        for spine in self.ax_setup.spines.values():
            spine.set_edgecolor('#FFA07A')
            spine.set_linewidth(3)

        self.ax_setup.tick_params(colors='#FFA07A', labelsize=8)

        # Solution Board area
        self.ax_solution.set_xlim(-0.5, self.board_width + 0.5)
        self.ax_solution.set_ylim(-0.5, self.board_height + 0.5)
        self.ax_solution.set_aspect('equal')
        self.ax_solution.set_title('SOLUTION', fontsize=16,
                                  fontweight='bold', color='#98D8C8', pad=15)
        self.ax_solution.set_facecolor('#16213e')

        for spine in self.ax_solution.spines.values():
            spine.set_edgecolor('#98D8C8')
            spine.set_linewidth(3)

        self.ax_solution.tick_params(colors='#98D8C8', labelsize=8)
    
    def _setup_buttons(self):
        """Create enhanced control buttons."""
        button_height = 0.045
        button_width = 0.065
        y_pos = 0.02
        
        # Button styling
        btn_style = {
            'rotate': {'color': '#4ECDC4', 'hovercolor': '#45B7D1'},
            'delete': {'color': '#E63946', 'hovercolor': '#FF6B6B'},
            'reset': {'color': '#FFA07A', 'hovercolor': '#FFB88C'},
            'solve': {'color': '#52B788', 'hovercolor': '#74C69D'},
            'nav': {'color': '#8338EC', 'hovercolor': '#9D4EDD'}
        }

        # Setup Board Buttons
        start_x = 0.32
        gap = 0.08

        ax_rotate = plt.axes([start_x, y_pos, button_width, button_height])
        self.btn_rotate = Button(ax_rotate, '⟲ Rotate',
                                 color=btn_style['rotate']['color'],
                                 hovercolor=btn_style['rotate']['hovercolor'])
        self.btn_rotate.label.set_fontsize(10)
        self.btn_rotate.label.set_fontweight('bold')
        self.btn_rotate.on_clicked(self._on_rotate_clicked)
        
        ax_remove = plt.axes([start_x + gap, y_pos, button_width, button_height])
        self.btn_remove = Button(ax_remove, '✕ Delete',
                                color=btn_style['delete']['color'],
                                hovercolor=btn_style['delete']['hovercolor'])
        self.btn_remove.label.set_fontsize(10)
        self.btn_remove.label.set_fontweight('bold')
        self.btn_remove.on_clicked(self._on_remove_clicked)
        
        ax_reset = plt.axes([start_x + gap * 2, y_pos, button_width, button_height])
        self.btn_reset = Button(ax_reset, '↺ Reset',
                               color=btn_style['reset']['color'],
                               hovercolor=btn_style['reset']['hovercolor'])
        self.btn_reset.label.set_fontsize(10)
        self.btn_reset.label.set_fontweight('bold')
        self.btn_reset.on_clicked(self._on_reset_clicked)
        
        ax_solve = plt.axes([start_x + gap * 3, y_pos, button_width, button_height])
        self.btn_solve = Button(ax_solve, '⚡ SOLVE',
                               color=btn_style['solve']['color'],
                               hovercolor=btn_style['solve']['hovercolor'])
        self.btn_solve.label.set_fontsize(11)
        self.btn_solve.label.set_fontweight('bold')
        self.btn_solve.on_clicked(self._on_solve_clicked)

        # Solution Navigation Buttons
        start_x_sol = 0.74

        ax_prev = plt.axes([start_x_sol, y_pos, button_width, button_height])
        self.btn_prev = Button(ax_prev, '◀ Prev',
                              color=btn_style['nav']['color'],
                              hovercolor=btn_style['nav']['hovercolor'])
        self.btn_prev.label.set_fontsize(10)
        self.btn_prev.label.set_fontweight('bold')
        self.btn_prev.on_clicked(self._on_prev_solution)

        ax_next = plt.axes([start_x_sol + gap, y_pos, button_width, button_height])
        self.btn_next = Button(ax_next, 'Next ▶',
                              color=btn_style['nav']['color'],
                              hovercolor=btn_style['nav']['hovercolor'])
        self.btn_next.label.set_fontsize(10)
        self.btn_next.label.set_fontweight('bold')
        self.btn_next.on_clicked(self._on_next_solution)

    def _connect_events(self):
        """Connect mouse and keyboard events."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _draw_all(self):
        """Redraw all pieces with enhanced styling."""
        for piece in self.pieces:
            for patch in piece.patches:
                patch.remove()
            piece.patches = []
        
        for piece in self.pieces:
            self._draw_piece(piece)
        
        self.fig.canvas.draw_idle()
    
    def _draw_piece(self, piece):
        """Draw a single piece with enhanced visual effects."""
        board_coords = piece.get_board_coords()
        ax = self.ax_setup if piece.placed or piece.dragging else self.ax_palette
        
        for x, y in board_coords:
            # Enhanced styling based on state
            if piece.invalid_flash > 0:
                # Invalid placement flash effect
                edge_color = '#FF0000'
                edge_width = 6
                alpha = 1.0
                zorder = 15
            elif piece.dragging:
                # Dragging state
                edge_color = '#00FF00'
                edge_width = 3.5
                alpha = 0.95
                zorder = 12
            elif piece.selected:
                edge_color = '#FFD700'
                edge_width = 4
                alpha = 1.0
                zorder = 10
            elif piece.placed:
                edge_color = '#FFFFFF'
                edge_width = 2.5
                alpha = 0.9
                zorder = 5
            else:
                edge_color = '#888888'
                edge_width = 2
                alpha = 0.85
                zorder = 1

            # Main piece rectangle with shadow effect
            if piece.selected or piece.dragging:
                shadow = patches.Rectangle(
                    (x + 0.05, y - 0.05), 0.95, 0.95,
                    linewidth=0,
                    facecolor='black',
                    alpha=0.3,
                    zorder=zorder - 1
                )
                ax.add_patch(shadow)
                piece.patches.append(shadow)
            
            rect = patches.Rectangle(
                (x, y), 1, 1,
                linewidth=edge_width,
                edgecolor=edge_color,
                facecolor=piece.color,
                alpha=alpha,
                zorder=zorder,
                picker=True
            )
            ax.add_patch(rect)
            piece.patches.append(rect)
            
            # Add piece number with enhanced styling
            text_color = '#000000' if sum(int(piece.color[i:i+2], 16)
                         for i in (1, 3, 5)) > 400 else '#FFFFFF'

            text = ax.text(
                x + 0.5, y + 0.5, str(piece.piece_index),
                ha='center', va='center',
                fontsize=13 if piece.selected or piece.dragging else 11,
                fontweight='bold',
                color=text_color,
                zorder=zorder + 1
            )
            piece.patches.append(text)
    
    def _find_piece_at(self, x, y, ax):
        """Find piece at given coordinates."""
        for piece in reversed(self.pieces):
            correct_ax = self.ax_setup if piece.placed or piece.dragging else self.ax_palette
            if correct_ax != ax:
                continue
            if piece.contains_point(x, y):
                return piece
        return None
    
    def _is_valid_placement(self, piece, exclude_piece=None):
        """Check if piece placement is valid (no overlap, within bounds)."""
        board_coords = piece.get_board_coords()
        
        # Check bounds
        for x, y in board_coords:
            if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
                return False
        
        # Check overlap with other placed pieces
        for other in self.pieces:
            if other == piece or other == exclude_piece or not other.placed:
                continue
            other_coords = set(other.get_board_coords())
            if any(coord in other_coords for coord in board_coords):
                return False
        
        return True
    
    def _on_mouse_press(self, event):
        """Handle mouse click - toggle dragging mode or place piece."""
        if event.inaxes not in [self.ax_palette, self.ax_setup]:
            return
        
        # If we have a piece being dragged, try to place it
        if self.selected_piece and self.selected_piece.dragging:
            self._try_place_piece()
            return
        
        # Otherwise, try to pick up a piece
        piece = self._find_piece_at(event.xdata, event.ydata, event.inaxes)
        
        if piece:
            # Deselect previous
            if self.selected_piece and self.selected_piece != piece:
                self.selected_piece.selected = False
                self.selected_piece.dragging = False
            
            # Select and start dragging new piece
            self.selected_piece = piece
            piece.selected = True
            piece.dragging = True
            
            # Store original position for ESC cancel
            piece.original_position = piece.position
            piece.original_placed = piece.placed
            
            # Calculate drag offset
            board_coords = piece.get_board_coords()
            if board_coords:
                avg_x = sum(x for x, y in board_coords) / len(board_coords)
                avg_y = sum(y for x, y in board_coords) / len(board_coords)
                self.drag_offset = (event.xdata - avg_x, event.ydata - avg_y)
            
            self._draw_all()
        else:
            # Deselect if clicking empty space
            if self.selected_piece:
                self.selected_piece.selected = False
                self.selected_piece.dragging = False
                self.selected_piece = None
                self._draw_all()
    
    def _on_mouse_move(self, event):
        """Handle mouse drag - only when piece is in dragging mode."""
        if not self.selected_piece or not self.selected_piece.dragging:
            return
        
        if event.inaxes not in [self.ax_palette, self.ax_setup]:
            return
        
        # Move piece to follow mouse
        self.selected_piece.move_to(
            event.xdata - self.drag_offset[0],
            event.ydata - self.drag_offset[1]
        )
        
        self._draw_all()
    
    def _on_mouse_release(self, event):
        """Mouse release no longer places pieces - only click or space does."""
        pass
    
    def _try_place_piece(self):
        """Try to place the currently dragging piece."""
        if not self.selected_piece or not self.selected_piece.dragging:
            return
        
        # Snap to grid
        board_coords = self.selected_piece.get_board_coords()
        if board_coords:
            avg_x = sum(x for x, y in board_coords) / len(board_coords)
            avg_y = sum(y for x, y in board_coords) / len(board_coords)
            snap_x = round(avg_x)
            snap_y = round(avg_y)
            
            offset_x = snap_x - avg_x
            offset_y = snap_y - avg_y
            
            self.selected_piece.position = (
                self.selected_piece.position[0] + int(offset_x),
                self.selected_piece.position[1] + int(offset_y)
            )
        
        # Check if position is on board
        board_coords = self.selected_piece.get_board_coords()
        is_on_board = all(
            0 <= x < self.board_width and 0 <= y < self.board_height
            for x, y in board_coords
        )
        
        if is_on_board:
            # Check if valid placement
            if self._is_valid_placement(self.selected_piece):
                # Valid placement - place piece
                self.selected_piece.placed = True
                self.selected_piece.dragging = False
                self.selected_piece.selected = False
                self.selected_piece = None
                self._draw_all()
            else:
                # Invalid placement - show flash effect
                self._show_invalid_flash()
        else:
            # Not on board - return to palette
            self._return_to_palette()
    
    def _show_invalid_flash(self):
        """Show visual feedback for invalid placement."""
        if not self.selected_piece:
            return
        
        self.selected_piece.invalid_flash = 3  # Flash 3 times
        self._animate_flash()
    
    def _animate_flash(self):
        """Animate the invalid placement flash."""
        if not self.selected_piece or self.selected_piece.invalid_flash <= 0:
            if self.selected_piece:
                self.selected_piece.invalid_flash = 0
                self._draw_all()
            return
        
        self.selected_piece.invalid_flash -= 1
        self._draw_all()
        
        # Schedule next flash
        self.flash_timer = self.fig.canvas.new_timer(interval=150)
        self.flash_timer.single_shot = True
        self.flash_timer.add_callback(self._animate_flash)
        self.flash_timer.start()
    
    def _return_to_palette(self):
        """Return dragging piece to palette."""
        if not self.selected_piece:
            return
        
        self.selected_piece.placed = False
        self.selected_piece.dragging = False
        self.selected_piece.selected = False
        
        # Find empty spot in palette
        palette_y = 0
        for p in self.pieces:
            if not p.placed and p != self.selected_piece:
                max_y = max(y for x, y in p.get_board_coords())
                palette_y = max(palette_y, max_y + 2)
        
        self.selected_piece.position = (-6, palette_y)
        self.selected_piece = None
        self._draw_all()
    
    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'r' and self.selected_piece:
            self._rotate_selected()
        elif event.key in ['delete', 'backspace'] and self.selected_piece:
            self._remove_selected()
        elif event.key == ' ' and self.selected_piece and self.selected_piece.dragging:
            # Space bar places piece
            self._try_place_piece()
        elif event.key == 'escape' and self.selected_piece and self.selected_piece.dragging:
            # ESC cancels dragging and returns to original position
            self._cancel_dragging()
    
    def _on_rotate_clicked(self, event):
        """Rotate button clicked."""
        self._rotate_selected()
    
    def _rotate_selected(self):
        """Rotate the selected piece."""
        if not self.selected_piece:
            return
        
        self.selected_piece.rotate()
        
        # If placed (not dragging), check if still valid
        if self.selected_piece.placed and not self.selected_piece.dragging:
            if not self._is_valid_placement(self.selected_piece):
                # Revert rotation
                for _ in range(3):
                    self.selected_piece.rotate()
        
        self._draw_all()
    
    def _cancel_dragging(self):
        """Cancel dragging and return piece to original position."""
        if not self.selected_piece or not self.selected_piece.dragging:
            return
        
        # Restore original position
        if hasattr(self.selected_piece, 'original_position'):
            self.selected_piece.position = self.selected_piece.original_position
            self.selected_piece.placed = self.selected_piece.original_placed
        else:
            # Fallback: return to palette
            self._return_to_palette()
            return
        
        self.selected_piece.dragging = False
        self.selected_piece.selected = False
        self.selected_piece = None
        self._draw_all()
    
    def _on_remove_clicked(self, event):
        """Remove button clicked."""
        self._remove_selected()
    
    def _remove_selected(self):
        """Remove selected piece from board."""
        if not self.selected_piece or not self.selected_piece.placed:
            return
        
        self.selected_piece.placed = False
        # Find empty spot in palette
        palette_y = 0
        for p in self.pieces:
            if not p.placed:
                max_y = max(y for x, y in p.get_board_coords())
                palette_y = max(palette_y, max_y + 2)
        self.selected_piece.position = (-6, palette_y)
        self._draw_all()
    
    def _on_reset_clicked(self, event):
        """Reset all pieces to palette."""
        for piece in self.pieces:
            piece.placed = False
            piece.selected = False
            piece.dragging = False
        
        # Reposition in palette
        palette_y = 0
        for piece in self.pieces:
            piece.position = (-6, palette_y)
            max_y = max(y for x, y in piece.coords)
            palette_y += max_y + 2
        
        self.selected_piece = None
        self._draw_all()
    
    def _on_solve_clicked(self, event):
        """Start solving with current configuration."""
        # Get fixed pieces
        fixed_pieces = []
        for piece in self.pieces:
            if piece.placed:
                fixed_pieces.append({
                    'index': piece.piece_index,
                    'coords': piece.get_board_coords()
                })
        
        print(f"\n{'='*60}")
        print(f"🔍 Starting solver with {len(fixed_pieces)} fixed pieces")
        for fp in fixed_pieces:
            print(f"  Piece #{fp['index']}: {fp['coords']}")
        print('='*60)

        self.ax_solution.clear()
        self.ax_solution.set_facecolor('#16213e')
        self.ax_solution.text(0.5, 0.5, '⚙️ SOLVING...',
                             transform=self.ax_solution.transAxes,
                             ha='center', va='center',
                             fontsize=20, fontweight='bold',
                             color='#FFD700')
        self.fig.canvas.draw()

        solver = BrainBlockSolver(self.board_width, self.board_height,
                                 self.original_pieces, fixed_pieces)

        self.solver_solutions = []
        try:
            solution_limit = 20
            for solution in islice(solver.solve(), solution_limit):
                grid = solver.create_solution_grid(solution)
                self.solver_solutions.append(grid)
                print(f"✓ Found solution {len(self.solver_solutions)}")
        except Exception as e:
            print(f"Error during solving: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"📊 Total solutions found: {len(self.solver_solutions)}")
        print('='*60)

        if not self.solver_solutions:
            self.ax_solution.clear()
            self.ax_solution.set_facecolor('#16213e')
            self.ax_solution.text(0.5, 0.5, 'NO SOLUTION',
                                 transform=self.ax_solution.transAxes,
                                 ha='center', va='center',
                                 fontsize=20, fontweight='bold',
                                 color='#E63946')
        else:
            self.current_solution_idx = 0
            self._display_current_solution()

        self.fig.canvas.draw()

    def _display_current_solution(self):
        """Display the solution with enhanced visuals."""
        self.ax_solution.clear()
        self.ax_solution.set_xlim(-0.5, self.board_width + 0.5)
        self.ax_solution.set_ylim(-0.5, self.board_height + 0.5)
        self.ax_solution.set_aspect('equal')
        self.ax_solution.set_facecolor('#0f3460')

        if not self.solver_solutions:
            return

        grid = self.solver_solutions[self.current_solution_idx]
        
        # Draw grid lines
        for x in range(self.board_width + 1):
            self.ax_solution.axvline(x, color='#98D8C8', linewidth=1, alpha=0.4)
        for y in range(self.board_height + 1):
            self.ax_solution.axhline(y, color='#98D8C8', linewidth=1, alpha=0.4)

        for x in range(self.board_width):
            for y in range(self.board_height):
                piece_idx = grid[x][y]
                if piece_idx >= 0:
                    color = PIECE_COLORS[piece_idx % len(PIECE_COLORS)]

                    is_fixed = False
                    for p in self.pieces:
                        if p.placed and p.piece_index == piece_idx and p.contains_point(x, y):
                            is_fixed = True
                            break

                    # Add glow effect for fixed pieces
                    if is_fixed:
                        glow = patches.Rectangle(
                            (x - 0.1, y - 0.1), 1.2, 1.2,
                            linewidth=0,
                            facecolor='#FFD700',
                            alpha=0.3,
                            zorder=1
                        )
                        self.ax_solution.add_patch(glow)

                    rect = patches.Rectangle(
                        (x, y), 1, 1,
                        linewidth=3 if is_fixed else 2,
                        edgecolor='#FFD700' if is_fixed else '#FFFFFF',
                        facecolor=color,
                        alpha=0.95,
                        zorder=5
                    )
                    self.ax_solution.add_patch(rect)

                    # Add piece number
                    text_color = '#000000' if sum(int(color[i:i+2], 16)
                                 for i in (1, 3, 5)) > 400 else '#FFFFFF'
                    self.ax_solution.text(
                        x + 0.5, y + 0.5, str(piece_idx),
                        ha='center', va='center',
                        fontsize=12,
                        fontweight='bold',
                        color=text_color,
                        zorder=10
                    )

        # Enhanced title
        title_text = f'Solution {self.current_solution_idx + 1} / {len(self.solver_solutions)}'
        self.ax_solution.set_title(title_text, fontsize=16,
                                   fontweight='bold', color='#98D8C8', pad=15)

        for spine in self.ax_solution.spines.values():
            spine.set_edgecolor('#98D8C8')
            spine.set_linewidth(3)

    def _on_prev_solution(self, event):
        if not self.solver_solutions:
            return
        self.current_solution_idx = (self.current_solution_idx - 1) % len(self.solver_solutions)
        self._display_current_solution()
        self.fig.canvas.draw()

    def _on_next_solution(self, event):
        if not self.solver_solutions:
            return
        self.current_solution_idx = (self.current_solution_idx + 1) % len(self.solver_solutions)
        self._display_current_solution()
        self.fig.canvas.draw()
    
    def show(self):
        """Display the GUI."""
        plt.show()


# ============================================================================
# DLX PUZZLE SOLVER
# ============================================================================

class BrainBlockSolver(dlx.DLX):
    """Solver for brain block puzzles using Dancing Links algorithm."""
    
    def __init__(self, board_width, board_height, pieces, fixed_pieces=None):
        self.board_width = board_width
        self.board_height = board_height
        self.pieces = pieces
        self.num_pieces = len(pieces)
        self.fixed_pieces = fixed_pieces or []
        
        # Initialize DLX columns
        columns = self._create_columns()
        column_dict = dict(columns)
        
        # Initialize base DLX structure
        dlx.DLX.__init__(self, [(name[0], dlx.DLX.PRIMARY) for name in columns])
        
        # Generate and add all possible placements
        self._generate_rows(column_dict)
    
    def _create_columns(self):
        """Create DLX columns for board cells and piece usage."""
        columns = []
        col_idx = 0
        
        # Get occupied cells from fixed pieces
        occupied_cells = set()
        for fp in self.fixed_pieces:
            for x, y in fp['coords']:
                occupied_cells.add((x, y))
        
        # Columns for each board cell (excluding fixed cells)
        for x in range(self.board_width):
            for y in range(self.board_height):
                if (x, y) not in occupied_cells:
                    columns.append((('cell', x, y), col_idx))
                    col_idx += 1
        
        # Columns for each piece (excluding fixed pieces)
        fixed_indices = set(fp['index'] for fp in self.fixed_pieces)
        for i in range(self.num_pieces):
            if i not in fixed_indices:
                columns.append((('piece', i), col_idx))
                col_idx += 1
        
        return columns
    
    def _generate_rows(self, column_dict):
        """Generate all possible piece placements as DLX rows."""
        fixed_indices = set(fp['index'] for fp in self.fixed_pieces)
        occupied_cells = set()
        for fp in self.fixed_pieces:
            for x, y in fp['coords']:
                occupied_cells.add((x, y))
        
        for piece_idx, piece in enumerate(self.pieces):
            # Skip fixed pieces
            if piece_idx in fixed_indices:
                continue
            
            placements = generate_placements(
                piece, self.board_width, self.board_height, piece_idx
            )
            
            for placement, p_idx in placements.items():
                # Skip placements that overlap with fixed pieces
                if any(cell in occupied_cells for cell in placement):
                    continue
                
                # Create column list for this placement
                cols = []
                valid = True
                for x, y in placement:
                    if ('cell', x, y) in column_dict:
                        cols.append(column_dict[('cell', x, y)])
                    else:
                        # This cell is occupied by a fixed piece
                        valid = False
                        break
                
                if valid and ('piece', p_idx) in column_dict:
                    cols.append(column_dict[('piece', p_idx)])
                    self.appendRow(cols, (placement, p_idx))
    
    def create_solution_grid(self, solution):
        """Convert DLX solution to 2D grid."""
        grid = [[-1] * self.board_height for _ in range(self.board_width)]
        
        # Place fixed pieces first
        for fp in self.fixed_pieces:
            for x, y in fp['coords']:
                grid[x][y] = fp['index']
        
        # Place solution pieces
        for row_idx in solution:
            placement, piece_idx = self.N[row_idx]
            for x, y in placement:
                grid[x][y] = piece_idx
        
        return grid


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    config = PUZZLE_SETS[PUZZLE_SET]
    board_width, board_height = config['board']
    pieces = config['pieces']
    
    print("=" * 70)
    print(f"🧩 BRAIN BLOCK PUZZLE SOLVER - Enhanced Edition")
    print("=" * 70)
    print(f"📋 Puzzle Set: {PUZZLE_SET}")
    print(f"📐 Board Size: {board_width} x {board_height}")
    print(f"🎯 Number of Pieces: {len(pieces)}")
    print("=" * 70)
    print("\nINSTRUCTIONS:")
    print("  🎨 Left Panel: Piece Palette")
    print("  🎯 Center Panel: Setup Board (drag pieces here)")
    print("  🧩 Right Panel: Solution Display")
    print("\n🎮 CONTROLS:")
    print("  • Click piece once to start dragging")
    print("  • Click again or press SPACE to place piece")
    print("  • Press ESC to cancel dragging and return piece")
    print("  • Press 'R' or click 'Rotate' to rotate selected piece")
    print("  • Press 'Delete' or click 'Delete' to remove piece from board")
    print("  • Click 'Reset' to return all pieces to palette")
    print("  • Click 'SOLVE' to find solutions")
    print("  • Use 'Prev/Next' buttons to browse through solutions")
    print("  • Invalid placements will show red flashing border")
    print("=" * 70)
    print()

    gui = PuzzleGUI(board_width, board_height, pieces)
    gui.show()


if __name__ == '__main__':
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    
    main()