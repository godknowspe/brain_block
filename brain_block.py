#!/usr/bin/env python
import sys
from functools import reduce
import dlx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from itertools import islice

# 定義拼圖形狀
SET = 3

if SET == 1:
    BOAD_W = 3
    BOAD_H = 2
    pieces = [
        [(0,0), (0,1), (1,0)],
        [(0,0), (1,0)],
        [(0,0)]
    ]
elif SET == 2:
    BOAD_W = 8
    BOAD_H = 5
    pieces = [
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,0), (0,1), (0,2), (0,3)],
        [(0,0), (0,1), (0,2), (0,3)],
        [(0,0), (0,1), (0,2), (1,0)],
        [(0,0), (0,1), (0,2), (1,0)],
        [(0,0), (1,0), (1,1), (2,0)],
        [(0,0), (1,0), (1,1), (2,0)],
        [(0,0), (1,0), (1,1), (2,1)],
        [(0,0), (1,0), (1,1), (2,1)],
    ]
elif SET == 3:
    BOAD_W = 10
    BOAD_H = 6
    pieces = [
        [(0,0), (0,1), (1,0), (1,1), (2,0)],
        [(0,0), (1,0), (2,0), (2,1), (2,2)],
        [(0,0), (1,0), (2,0), (0,1), (2,1)],
        [(0,0), (1,0), (1,1), (1,2), (2,2)],
        [(0,1), (1,0), (1,1), (1,2), (2,1)],
        [(0,0), (1,0), (2,0), (3,0), (1,1)],
        [(0,0), (1,0), (2,0), (1,1), (1,2)],
        [(0,0), (1,0), (2,0), (3,0), (4,0)],
        [(0,0), (1,0), (1,1), (1,2), (2,1)],
        [(0,0), (1,0), (1,1), (2,1), (2,2)],
        [(0,0), (1,0), (2,0), (2,1), (3,1)],
        [(0,0), (1,0), (2,0), (3,0), (0,1)],
    ]

def move(shape, dx=0, dy=0):
    return tuple((x + dx, y + dy) for x, y in shape)

def normalize(coords):
    """將圖形平移到左下角 (0,0) 開始，並排序，方便比較是否相同"""
    min_x = min(x for x, y in coords)
    min_y = min(y for x, y in coords)
    shifted = [(x - min_x, y - min_y) for x, y in coords]
    return tuple(sorted(shifted))

def rotate(coords, angle):
    """旋轉座標 (angle = 0, 90, 180, 270)"""
    if angle == 0:
        return [(x, y) for x, y in coords]
    elif angle == 90:
        return [(-y, x) for x, y in coords]
    elif angle == 180:
        return [(-x, -y) for x, y in coords]
    elif angle == 270:
        return [(y, -x) for x, y in coords]

def mirror(coords):
    """鏡像翻轉 (左右翻轉)"""
    return [(-x, y) for x, y in coords]

def all_transformations(coords, dimx, dimy, p_index):
    """生成旋轉 + 鏡像的所有結果，並去重"""
    results = set()
    for flip in [False, True]:
        shape = mirror(coords) if flip else coords
        for angle in [0, 90, 180, 270]:
            r = rotate(shape, angle)
            init_loc = normalize(r)
            max_x = max(x for x, y in init_loc)
            max_y = max(y for x, y in init_loc)
            for i in range(dimx - max_x):
                for j in range(dimy - max_y):
                    ret = move(init_loc, i, j)
                    ret += (p_index,)
                    #print(ret, i, j)
                    results.add(move(init_loc, i, j))

    return {tuple(c): p_index for c in results}

def visualize(coords):
    """將座標轉成 ASCII 圖形"""
    #norm = normalize(coords)
    norm = coords
    max_x = max(x for x, y in norm)
    max_y = max(y for x, y in norm)
    grid = [["." for _ in range(max_x+1)] for _ in range(max_y+1)]
    for x, y in norm:
        grid[max_y-y][x] = "#"  # y 軸向上
    return "\n".join("".join(row) for row in grid)


class DLXbrainblock(dlx.DLX):
    def __init__(self, dimx, dimy):
        # Create the columns.
        ctr = 0
        cols = []
        self.num_pieces = len(pieces)
        self.dimx = dimx
        self.dimy = dimy
        self.ax = ''
        self.fig = ''
        self.sol_rec = []
        self.cur_sol = 0
        self.sol_start = 0

        for i in range(len(pieces)):
            print("Pieces #" + str(i))
            print(visualize(pieces[i]))
            print()

        # GENERATE x * y constraints: 3 * 2
        # Create the entry coverage, which determines that entry i,j in the grid is occupied.
        for x in range(self.dimx):
            cols += [(('e', x, y), ctr + y) for y in range(self.dimy)]
            ctr += self.dimy

        # Create the puzzle number, which determines that puzzle i is used.
        for i in range(self.num_pieces):
            cols += [(('p', i), ctr + i)]

        # Create a dictionary from this, which maps column name to column index.
        sdict = dict(cols)
        #print('Dic', sdict)
        
        # Create the DLX object.
        dlx.DLX.__init__(self, [(colname[0], dlx.DLX.PRIMARY) for colname in cols])

        # GENERATE all options, add options to DLX by appendRow
        # Now create all possible rows.
        rowdict = {} # Store rowindex which is returned by appendRow
        self.lookupdict = {}

        transformed = [all_transformations(pieces[i], dimx, dimy, i) for i in range(len(pieces))]
        #print("Option set", transformed)

        for set in transformed:
            for s,p in set.items():
                # option UID = appendRow(UID list, lookup key)
                #print("S", s, "p", p)
                for x, y in s:
                    keyl = [sdict[('e', x, y)] for x, y in s]
                keyl.append(sdict[('p', p)])
                #print(keyl)
                val =  self.appendRow(keyl, (s,p))
                rowdict[(s,p)] = val
                """
                print(visualize(s))
                print()
                """

    def createSolutionGrid(self, sol):
        '''Return a two dimensional grid representing the solution.'''

        # We need to determine what is represented by each row. This is easily accessed by rowname.
        solgrid = [['#']*self.dimy for i in range(self.dimx)]
        for a in sol:
            s,p = self.N[a]
            #print (s, p)
            for x,y in s:
                """
                print('x', x)
                print('y', y)
                print('p', p)
                """
                solgrid[x][y] = p
        return solgrid

    def createSolutionGridString(self, sol):
        '''Create a string representing the solution grid in nice format.'''

        grid = self.createSolutionGrid(sol)

        s = ""
        for row in reversed(list(zip(*grid))):
            s += " ".join(map(str, row)) + "\n"

        return s

    def visualize_solution(self, sol, sol_num):
        solgrid = [['#']*self.dimy for i in range(self.dimx)]
        for a in sol:
            s,p = self.N[a]
            for x,y in s:
                solgrid[x][y] = p

        cmap = plt.get_cmap("tab20")
        recl = []
        for y in range(self.dimy):
            for x in range(self.dimx):
                rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5,
                                         edgecolor='gray', facecolor=tuple(x for x in cmap(solgrid[x][y]/20)[:3]))
                self.ax.add_patch(rect)
                if len(self.sol_rec) == 0:
                    rect.set_visible(True)
                else:
                    rect.set_visible(False)
                recl.append(rect)
                """
                ax.text(x + 0.5, y + 0.5, solgrid[x][y],
                        ha='center', va='center', fontsize=12, color='black')
                """

        if len(self.sol_rec) == 0:
            plt.title(f"Solution #"+str(sol_num))
            self.cur_sol = 0
            self.sol_start = sol_num
        self.sol_rec.append(recl)
        plt.show(block=False)

    def next_patch(self, event):
        for rec in self.sol_rec[self.cur_sol]:
            rec.set_visible(False)
        self.cur_sol = (self.cur_sol + 1) % len(self.sol_rec)
        for rec in self.sol_rec[self.cur_sol]:
            rec.set_visible(True)
        plt.title(f"Solution #"+str(self.sol_start + self.cur_sol))
        self.fig.canvas.draw_idle()

    def prev_patch(self, event):
        for rec in self.sol_rec[self.cur_sol]:
            rec.set_visible(False)
        self.cur_sol = (self.cur_sol - 1) % len(self.sol_rec)
        for rec in self.sol_rec[self.cur_sol]:
            rec.set_visible(True)
        plt.title(f"Solution #"+str(self.sol_start + self.cur_sol))
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass

    sol_start = 1
    sol_end = 10
    if len(sys.argv) == 3:
        sol_start = int(sys.argv[1])
        sol_end = int(sys.argv[2])

    d = DLXbrainblock(BOAD_W, BOAD_H)
    sol_cnt = 0

    plt.ion() # 開啟互動模式
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, d.dimx)
    ax.set_ylim(0, d.dimy)
    ax.set_aspect('equal')
    ax.axis('off')
    d.ax = ax
    d.fig = fig

    # 加按鈕
    axprev = plt.axes([0.4, 0.9, 0.08, 0.025])
    axnext = plt.axes([0.5, 0.9, 0.08, 0.025])
    bprev = Button(axprev, 'Previous')
    bnext = Button(axnext, 'Next')
    bprev.on_clicked(d.prev_patch)
    bnext.on_clicked(d.next_patch)

    for sol in islice(d.solve(), sol_end):
        sol_cnt+=1
        #print(sol)
        
        #print('SOLUTION #' + str(sol_cnt) + ':')
        #print(d.createSolutionGridString(sol))

        if sol_cnt >= sol_start and sol_cnt <= sol_end:
            d.visualize_solution(sol, sol_cnt)

    print('Total solution cnt:', sol_cnt)

    # 等待視窗關閉才結束程式
    plt.ioff()
    plt.show()
    