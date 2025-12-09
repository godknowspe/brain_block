#!/usr/bin/env python
"""
Brain Block Puzzle Solver with Interactive GUI
Allows users to place initial pieces before solving
"""

import sys
from itertools import islice
import dlx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np

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
# INTERACTIVE GUI
# ============================================================================

class PuzzleGUI:
    """Interactive GUI for setting up initial puzzle state."""
    
    def __init__(self, board_width, board_height, pieces):
        self.board_width = board_width
        self.board_height = board_height
        self.original_pieces = pieces
        self.cmap = plt.get_cmap('tab20')
        
        # Create interactive pieces
        self.pieces = []
        palette_y = 0
        for i, piece_coords in enumerate(pieces):
            color = self.cmap(i / 20)[:3]
            pos = (-6, palette_y)  # Position in palette area
            self.pieces.append(InteractivePiece(piece_coords, i, color, pos))
            
            # Stack pieces vertically in palette
            max_y = max(y for x, y in normalize(piece_coords))
            palette_y += max_y + 2
        
        self.selected_piece = None
        self.drag_offset = (0, 0)

        # Solution state
        self.solver_solutions = []
        self.solution_patches = []
        self.current_solution_idx = 0
        
        # Setup matplotlib figure
        # Split into 3 columns: Palette | Setup Board | Solution Board
        self.fig, (self.ax_palette, self.ax_setup, self.ax_solution) = plt.subplots(
            1, 3, figsize=(18, 8),
            gridspec_kw={'width_ratios': [0.5, 1.25, 1.25]}
        )
        
        self._setup_axes()
        self._setup_buttons()
        self._connect_events()
        self._draw_all()
        
    
    def _setup_axes(self):
        """Configure plot axes."""
        # Palette area (left)
        self.ax_palette.set_xlim(-8, 2)
        self.ax_palette.set_ylim(-2, 50)
        self.ax_palette.set_aspect('equal')
        self.ax_palette.set_title('Palette', fontsize=12, pad=10)
        self.ax_palette.grid(True, alpha=0.3)
        self.ax_palette.set_axis_off() # Hide axis ticks for cleaner look
        
        # Setup Board area (center)
        self.ax_setup.set_xlim(0, self.board_width)
        self.ax_setup.set_ylim(0, self.board_height)
        self.ax_setup.set_aspect('equal')
        self.ax_setup.set_title('Setup (Place Pieces)', fontsize=14, pad=10)
        self.ax_setup.grid(True, alpha=0.3)
        self.ax_setup.invert_yaxis() # Usually puzzles are top-left origin?
        # Wait, the original code didn't invert y axis. Let's keep it consistent.
        # Original code used default y axis (bottom-up).
        # Let's check: set_ylim(0, height). Matplotlib default is 0 at bottom.
        
        # Draw board grid for Setup
        for x in range(self.board_width + 1):
            self.ax_setup.axvline(x, color='black', linewidth=2)
        for y in range(self.board_height + 1):
            self.ax_setup.axhline(y, color='black', linewidth=2)

        # Solution Board area (right)
        self.ax_solution.set_xlim(0, self.board_width)
        self.ax_solution.set_ylim(0, self.board_height)
        self.ax_solution.set_aspect('equal')
        self.ax_solution.set_title('Solution Result', fontsize=14, pad=10)
        self.ax_solution.axis('off') # Hide axis for cleaner solution view
    
    def _setup_buttons(self):
        """Create control buttons."""
        button_height = 0.04
        button_width = 0.06
        y_pos = 0.02
        
        # Setup Board Buttons (Center area roughly)
        # Assuming Setup is roughly 0.2 to 0.55 in figure coords
        start_x_setup = 0.25
        gap = 0.07

        ax_rotate = plt.axes([start_x_setup, y_pos, button_width, button_height])
        self.btn_rotate = Button(ax_rotate, 'Rotate')
        self.btn_rotate.on_clicked(self._on_rotate_clicked)
        
        ax_remove = plt.axes([start_x_setup + gap, y_pos, button_width, button_height])
        self.btn_remove = Button(ax_remove, 'Delete')
        self.btn_remove.on_clicked(self._on_remove_clicked)
        
        ax_reset = plt.axes([start_x_setup + gap * 2, y_pos, button_width, button_height])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset_clicked)
        
        ax_solve = plt.axes([start_x_setup + gap * 3, y_pos, button_width, button_height])
        self.btn_solve = Button(ax_solve, 'Solve', color='lightgreen')
        self.btn_solve.on_clicked(self._on_solve_clicked)

        # Solution Board Buttons (Right area)
        start_x_sol = 0.70

        ax_prev = plt.axes([start_x_sol, y_pos, button_width, button_height])
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_prev.on_clicked(self._on_prev_solution)

        ax_next = plt.axes([start_x_sol + gap, y_pos, button_width, button_height])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self._on_next_solution)

    def _connect_events(self):
        """Connect mouse and keyboard events."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _draw_all(self):
        """Redraw all pieces."""
        # Clear existing patches on Setup/Palette
        for piece in self.pieces:
            for patch in piece.patches:
                patch.remove()
            piece.patches = []
        
        # Draw pieces
        for piece in self.pieces:
            self._draw_piece(piece)
        
        self.fig.canvas.draw_idle()
    
    def _draw_piece(self, piece):
        """Draw a single piece."""
        board_coords = piece.get_board_coords()
        
        # Determine which axis to draw on
        ax = self.ax_setup if piece.placed else self.ax_palette
        
        for x, y in board_coords:
            # Determine edge color based on state
            if piece.selected:
                edge_color = 'red'
                linewidth = 3
            elif piece.placed:
                edge_color = 'darkblue'
                linewidth = 2
            else:
                edge_color = 'gray'
                linewidth = 1.5
            
            rect = patches.Rectangle(
                (x, y), 1, 1,
                linewidth=linewidth,
                edgecolor=edge_color,
                facecolor=piece.color,
                alpha=0.8 if not piece.selected else 1.0,
                picker=True
            )
            ax.add_patch(rect)
            piece.patches.append(rect)
            
            # Add piece number
            text = ax.text(
                x + 0.5, y + 0.5, str(piece.piece_index),
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white'
            )
            piece.patches.append(text)
    
    def _find_piece_at(self, x, y, ax):
        """Find piece at given coordinates."""
        # Check in reverse order (topmost first)
        for piece in reversed(self.pieces):
            correct_ax = self.ax_setup if piece.placed else self.ax_palette
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
        """Handle mouse click."""
        if event.inaxes not in [self.ax_palette, self.ax_setup]:
            return
        
        piece = self._find_piece_at(event.xdata, event.ydata, event.inaxes)
        
        if piece:
            # Deselect previous
            if self.selected_piece and self.selected_piece != piece:
                self.selected_piece.selected = False
            
            # Select new piece
            self.selected_piece = piece
            piece.selected = True
            
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
                self.selected_piece = None
                self._draw_all()
    
    def _on_mouse_move(self, event):
        """Handle mouse drag."""
        if not self.selected_piece or event.inaxes not in [self.ax_palette, self.ax_setup]:
            return
        
        # Move piece to follow mouse
        self.selected_piece.move_to(
            event.xdata - self.drag_offset[0],
            event.ydata - self.drag_offset[1]
        )
        
        # Update placed status based on which axis
        self.selected_piece.placed = (event.inaxes == self.ax_setup)
        
        self._draw_all()
    
    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if not self.selected_piece:
            return
        
        # If on board, snap to grid and validate
        if self.selected_piece.placed:
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
            
            # Check if valid
            if not self._is_valid_placement(self.selected_piece):
                # Return to palette
                self.selected_piece.placed = False
                # Find empty spot in palette
                palette_y = 0
                for p in self.pieces:
                    if not p.placed and p != self.selected_piece:
                        max_y = max(y for x, y in p.get_board_coords())
                        palette_y = max(palette_y, max_y + 2)
                self.selected_piece.position = (-6, palette_y)
        
        # Deselect piece to stop dragging
        self.selected_piece.selected = False
        self.selected_piece = None

        self._draw_all()
    
    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'r' and self.selected_piece:
            self._rotate_selected()
        elif event.key in ['delete', 'backspace'] and self.selected_piece:
            self._remove_selected()
    
    def _on_rotate_clicked(self, event):
        """Rotate button clicked."""
        self._rotate_selected()
    
    def _rotate_selected(self):
        """Rotate the selected piece."""
        if not self.selected_piece:
            return
        
        self.selected_piece.rotate()
        
        # If on board, check if still valid
        if self.selected_piece.placed:
            if not self._is_valid_placement(self.selected_piece):
                # Revert rotation
                for _ in range(3):
                    self.selected_piece.rotate()
        
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
        
        print(f"\nStarting solver with {len(fixed_pieces)} fixed pieces")
        for fp in fixed_pieces:
            print(f"  Piece #{fp['index']}: {fp['coords']}")

        # Clear previous solutions
        self.ax_solution.clear()
        self.ax_solution.set_title("Solving...", fontsize=14, pad=10)
        self.ax_solution.axis('off')
        self.fig.canvas.draw()

        # Run solver (search for first 20 solutions)
        solver = BrainBlockSolver(self.board_width, self.board_height, self.original_pieces, fixed_pieces)

        self.solver_solutions = []
        try:
            solution_limit = 20
            for solution in islice(solver.solve(), solution_limit):
                # Store necessary data to reconstruct solution grid later
                # BrainBlockSolver.visualize_solution requires reconstructing the grid
                # So we can just store the DLX result (list of row indices)
                # And use the solver instance (which holds the mappings) to interpret it
                # However, solver instance is local. We should probably keep it or helper methods.

                # Let's convert the solution to a grid immediately to store it
                grid = solver.create_solution_grid(solution)
                self.solver_solutions.append(grid)
                print(f"Found solution {len(self.solver_solutions)}")
        except Exception as e:
            print(f"Error during solving: {e}")
            import traceback
            traceback.print_exc()

        print(f"Total solutions found: {len(self.solver_solutions)}")

        if not self.solver_solutions:
            self.ax_solution.set_title("No Solution Found", fontsize=14, pad=10)
            self.ax_solution.axis('off')
        else:
            self.current_solution_idx = 0
            self._display_current_solution()

        self.fig.canvas.draw()

    def _display_current_solution(self):
        """Display the solution at current_solution_idx on ax_solution."""
        self.ax_solution.clear()
        self.ax_solution.axis('off')
        self.ax_solution.set_xlim(0, self.board_width)
        self.ax_solution.set_ylim(0, self.board_height)
        self.ax_solution.set_aspect('equal')

        if not self.solver_solutions:
            return

        grid = self.solver_solutions[self.current_solution_idx]
        
        # Reuse pieces fixed pieces logic? No, just draw the grid.
        # Check if we have fixed pieces in this grid?
        # The grid contains piece indices.

        for x in range(self.board_width):
            for y in range(self.board_height):
                piece_idx = grid[x][y]
                if piece_idx >= 0:
                    color = self.cmap(piece_idx / 20)[:3]

                    # Check if this cell is part of a fixed piece from current GUI state?
                    # We can iterate through self.pieces and see if they are placed and cover (x,y)
                    is_fixed = False
                    for p in self.pieces:
                        if p.placed and p.piece_index == piece_idx and p.contains_point(x, y):
                             is_fixed = True
                             break

                    # Flip y-coordinate to match setup board orientation
                    display_y = self.board_height - 1 - y

                    rect = patches.Rectangle(
                        (x, display_y), 1, 1,
                        linewidth=2.5 if is_fixed else 1.5,
                        edgecolor='darkred' if is_fixed else 'black',
                        facecolor=color,
                        alpha=0.9 if is_fixed else 0.7
                    )
                    self.ax_solution.add_patch(rect)

        self.ax_solution.set_title(f'Solution {self.current_solution_idx + 1} / {len(self.solver_solutions)}', fontsize=14, pad=10)

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
    # Get puzzle configuration
    config = PUZZLE_SETS[PUZZLE_SET]
    board_width, board_height = config['board']
    pieces = config['pieces']
    
    print("=" * 60)
    print(f"Brain Block Solver - Puzzle Set {PUZZLE_SET}")
    print(f"Board Size: {board_width} x {board_height}")
    print(f"Number of Pieces: {len(pieces)}")
    print("=" * 60)
    print("\nInstructions:")
    print("  - Left Panel: Palette")
    print("  - Center Panel: Setup Board. Drag pieces here.")
    print("  - Right Panel: Solution Result.")
    print("  - Buttons allow Rotate, Delete, Reset, and Solve.")
    print("  - After solving, use Prev/Next to view solutions.")
    print("  - You can modify the setup and Solve again.\n")
    
    # Show interactive GUI
    gui = PuzzleGUI(board_width, board_height, pieces)
    gui.show()


if __name__ == '__main__':
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    
    main()
