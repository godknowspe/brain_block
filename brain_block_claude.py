#!/usr/bin/env python
"""
Brain Block Puzzle Solver using Dancing Links (DLX) Algorithm
Solves polyomino tiling puzzles with visualization
"""

import sys
from itertools import islice
import dlx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

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

# Select puzzle set
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
        
        # Generate 4 rotations
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
        
        # Try all positions where piece fits on board
        for dx in range(board_width - max_x):
            for dy in range(board_height - max_y):
                placement = translate(orientation, dx, dy)
                placements[placement] = piece_index
    
    return placements


def visualize_piece(coords):
    """Convert coordinates to ASCII art for console display."""
    if not coords:
        return ""
    
    norm = normalize(coords)
    max_x = max(x for x, y in norm)
    max_y = max(y for x, y in norm)
    
    grid = [['.' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    
    for x, y in norm:
        grid[max_y - y][x] = '#'  # Y-axis points up
    
    return '\n'.join(''.join(row) for row in grid)


# ============================================================================
# DLX PUZZLE SOLVER
# ============================================================================

class BrainBlockSolver(dlx.DLX):
    """Solver for brain block puzzles using Dancing Links algorithm."""
    
    def __init__(self, board_width, board_height, pieces):
        self.board_width = board_width
        self.board_height = board_height
        self.pieces = pieces
        self.num_pieces = len(pieces)
        
        # Visualization state
        self.fig = None
        self.ax = None
        self.solution_patches = []
        self.current_solution_idx = 0
        self.solution_start_num = 0
        
        # Print puzzle pieces
        print("Puzzle Pieces:")
        print("=" * 50)
        for i, piece in enumerate(pieces):
            print(f"Piece #{i}:")
            print(visualize_piece(piece))
            print()
        
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
        
        # Columns for each board cell
        for x in range(self.board_width):
            for y in range(self.board_height):
                columns.append((('cell', x, y), col_idx))
                col_idx += 1
        
        # Columns for each piece (ensures each piece used exactly once)
        for i in range(self.num_pieces):
            columns.append((('piece', i), col_idx))
            col_idx += 1
        
        return columns
    
    def _generate_rows(self, column_dict):
        """Generate all possible piece placements as DLX rows."""
        for piece_idx, piece in enumerate(self.pieces):
            placements = generate_placements(
                piece, self.board_width, self.board_height, piece_idx
            )
            
            for placement, p_idx in placements.items():
                # Create column list for this placement
                cols = [column_dict[('cell', x, y)] for x, y in placement]
                cols.append(column_dict[('piece', p_idx)])
                
                # Add row to DLX
                self.appendRow(cols, (placement, p_idx))
    
    def create_solution_grid(self, solution):
        """Convert DLX solution to 2D grid."""
        grid = [[-1] * self.board_height for _ in range(self.board_width)]
        
        for row_idx in solution:
            placement, piece_idx = self.N[row_idx]
            for x, y in placement:
                grid[x][y] = piece_idx
        
        return grid
    
    def format_solution(self, solution):
        """Format solution as string for console output."""
        grid = self.create_solution_grid(solution)
        lines = []
        
        for y in range(self.board_height - 1, -1, -1):
            line = ' '.join(f'{grid[x][y]:2d}' for x in range(self.board_width))
            lines.append(line)
        
        return '\n'.join(lines)
    
    def visualize_solution(self, solution, solution_num):
        """Visualize solution using matplotlib."""
        grid = self.create_solution_grid(solution)
        cmap = plt.get_cmap('tab20')
        
        patches_list = []
        for x in range(self.board_width):
            for y in range(self.board_height):
                piece_idx = grid[x][y]
                color = cmap(piece_idx / 20)[:3] if piece_idx >= 0 else (1, 1, 1)
                
                rect = patches.Rectangle(
                    (x, y), 1, 1,
                    linewidth=1.5,
                    edgecolor='black',
                    facecolor=color
                )
                self.ax.add_patch(rect)
                
                # Hide all but first solution initially
                rect.set_visible(len(self.solution_patches) == 0)
                patches_list.append(rect)
        
        # Update title and state on first solution
        if len(self.solution_patches) == 0:
            plt.title(f'Solution #{solution_num}')
            self.current_solution_idx = 0
            self.solution_start_num = solution_num
        
        self.solution_patches.append(patches_list)
        plt.show(block=False)
    
    def show_next_solution(self, event):
        """Button callback to show next solution."""
        self._switch_solution(1)
    
    def show_previous_solution(self, event):
        """Button callback to show previous solution."""
        self._switch_solution(-1)
    
    def _switch_solution(self, direction):
        """Switch between solutions."""
        if not self.solution_patches:
            return
        
        # Hide current solution
        for patch in self.solution_patches[self.current_solution_idx]:
            patch.set_visible(False)
        
        # Update index
        self.current_solution_idx = (
            (self.current_solution_idx + direction) % len(self.solution_patches)
        )
        
        # Show new solution
        for patch in self.solution_patches[self.current_solution_idx]:
            patch.set_visible(True)
        
        # Update title
        solution_num = self.solution_start_num + self.current_solution_idx
        plt.title(f'Solution #{solution_num}')
        self.fig.canvas.draw_idle()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    # Parse command line arguments
    solution_start = 1
    solution_end = 10
    
    if len(sys.argv) == 3:
        try:
            solution_start = int(sys.argv[1])
            solution_end = int(sys.argv[2])
        except ValueError:
            print("Usage: python brain_block.py [start_solution] [end_solution]")
            sys.exit(1)
    
    # Get puzzle configuration
    config = PUZZLE_SETS[PUZZLE_SET]
    board_width, board_height = config['board']
    pieces = config['pieces']
    
    print(f"Solving Puzzle Set {PUZZLE_SET}")
    print(f"Board: {board_width}x{board_height}")
    print(f"Number of pieces: {len(pieces)}")
    print()
    
    # Create solver
    solver = BrainBlockSolver(board_width, board_height, pieces)
    
    # Setup visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, board_width)
    ax.set_ylim(0, board_height)
    ax.set_aspect('equal')
    ax.axis('off')
    
    solver.fig = fig
    solver.ax = ax
    
    # Add navigation buttons
    ax_prev = plt.axes([0.4, 0.92, 0.08, 0.04])
    ax_next = plt.axes([0.52, 0.92, 0.08, 0.04])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    btn_prev.on_clicked(solver.show_previous_solution)
    btn_next.on_clicked(solver.show_next_solution)
    
    # Find and display solutions
    solution_count = 0
    print("Searching for solutions...")
    
    for solution in islice(solver.solve(), solution_end):
        solution_count += 1
        
        if solution_start <= solution_count <= solution_end:
            solver.visualize_solution(solution, solution_count)
        
        print(f'\rSolutions found: {solution_count}', end='', flush=True)
    
    print(f'\n\nTotal solutions found: {solution_count}')
    print(f'Displaying solutions {solution_start} to {min(solution_count, solution_end)}')
    
    # Keep window open
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    
    main()
