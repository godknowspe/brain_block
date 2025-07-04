import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dlx

# 板子大小
BOARD_W, BOARD_H = 10, 6
BOARD_SIZE = BOARD_W * BOARD_H

# 定義拼圖形狀（12塊）
pieces = [
    [(0,0), (1,0), (2,0), (2,1)],          
    [(0,0), (1,0), (1,1), (2,1)],          
    [(0,1), (1,1), (1,0), (2,0)],          
    [(0,0), (0,1), (0,2), (1,1)],          
    [(0,0), (1,0), (2,0), (3,0)],          
    [(0,0), (1,0), (2,0), (1,1)],          
    [(0,1), (1,1), (2,1), (2,0)],          
    [(0,0), (1,0), (1,1), (1,2)],          
    [(0,0), (0,1), (1,1), (1,2)],          
    [(1,0), (0,1), (1,1), (2,1), (1,2)],   
    [(0,0), (1,0), (1,1), (2,1)],          
    [(0,0), (1,0), (2,0), (2,-1)]          
]

# 生成旋轉與鏡像的變化
def generate_orientations(shape):
    variants = set()
    for _ in range(4):
        shape = [(x, -y) for y, x in shape]
        for flip in [1, -1]:
            flipped = [(y, x * flip) for y, x in shape]
            min_y = min(pt[0] for pt in flipped)
            min_x = min(pt[1] for pt in flipped)
            norm = tuple(sorted((y - min_y, x - min_x) for y, x in flipped))
            variants.add(norm)
    return list(variants)

# 建立 DLX 矩陣
matrix = []
placements = []

matrix_test = [
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1]
]

for pid, piece in enumerate(pieces):
    for variant in generate_orientations(piece):
        max_y = max(y for y, x in variant)
        max_x = max(x for y, x in variant)
        for i in range(BOARD_H - max_y):
            for j in range(BOARD_W - max_x):
                cells = []
                for y, x in variant:
                    idx = (i + y) * BOARD_W + (j + x)
                    cells.append(idx)
                if len(set(cells)) == len(variant):
                    matrix.append([1 if i in cells else 0 for i in range(BOARD_SIZE)])
                    placements.append((pid, [(i + y, j + x) for y, x in variant]))

# 方法 1: 轉換為字典格式
def matrix_to_dlx_format(matrix):
    rows = []
    for i, row in enumerate(matrix):
        row_data = []
        for j, val in enumerate(row):
            if val == 1:
                row_data.append(j)  # 只記錄為1的列索引
        if row_data:  # 只添加非空行
            rows.append(row_data)
    return rows

# 轉換矩陣格式
#dlx_matrix = matrix_to_dlx_format(matrix_test)
#dlx_matrix = matrix_to_dlx_format(matrix)
#print("轉換後的格式:", dlx_matrix)

# 可能需要提供列名
columns = [(f'col_{i}', False) for i in range(len(matrix_test[0]))]

solver = dlx.DLX(columns)
# 然後添加行
for i, row in enumerate(matrix_test):
    row_data = [f'col_{j}' for j, val in enumerate(row) if val == 1]
    if row_data:
        solver.appendRow(row_data)

solutions = list(solver.solve())
print("解:", solutions)

# 視覺化指定解
def visualize_solution(solution_idx):
    sol = solutions[solution_idx]
    board = [['' for _ in range(BOARD_W)] for _ in range(BOARD_H)]
    for row_idx in sol:
        pid, positions = placements[row_idx]
        label = chr(ord('A') + pid)
        for y, x in positions:
            board[y][x] = label

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, BOARD_W)
    ax.set_ylim(0, BOARD_H)
    ax.set_aspect('equal')
    ax.axis('off')

    for y in range(BOARD_H):
        for x in range(BOARD_W):
            rect = patches.Rectangle((x, BOARD_H - y - 1), 1, 1, linewidth=0.5,
                                     edgecolor='gray', facecolor='lightblue' if board[y][x] else 'white')
            ax.add_patch(rect)
            if board[y][x]:
                ax.text(x + 0.5, BOARD_H - y - 0.5, board[y][x],
                        ha='center', va='center', fontsize=12, color='black')

    plt.title(f"Solution #{solution_idx + 1}")
    plt.show()

print(f"共找到 {len(solutions)} 組解法")
visualize_solution(0)  # 顯示第一組解
