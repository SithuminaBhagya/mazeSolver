import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------
# 1) Build maze_matrix (0=wall, 1=path) from threshold & cropping
# ---------------------------------------------------------------------

image_path = 'maze.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Define corners for cropping


corners = [
    (3, 2),
    (3, binary.shape[1] - 3),
    (binary.shape[0] - 2, 2),
    (binary.shape[0] - 2, binary.shape[1] - 3)
]
min_y = min(pt[0] for pt in corners)
max_y = max(pt[0] for pt in corners)
min_x = min(pt[1] for pt in corners)
max_x = max(pt[1] for pt in corners)

maze_height = max_y - min_y + 1
maze_width  = max_x - min_x + 1

maze_matrix = np.zeros((maze_height, maze_width), dtype=int)

for ry in range(min_y, max_y + 1):
    for rx in range(min_x, max_x + 1):
        if binary[ry, rx] == 255:
            maze_matrix[ry - min_y, rx - min_x] = 1  # wall
        else:
            maze_matrix[ry - min_y, rx - min_x] = 0  # path

# ---------------------------------------------------------------------
# 2) Skip partial boxes: only consider full 20×20 boxes
# ---------------------------------------------------------------------
BOX_SIZE = 20
box_rows = maze_height // BOX_SIZE  # excludes partial leftover at bottom
box_cols = maze_width // BOX_SIZE   # excludes partial leftover at right

# ---------------------------------------------------------------------
# 3) Find the start and end cells
# ---------------------------------------------------------------------
top_row = maze_matrix[0, :]  
lowest_full_row = box_rows * BOX_SIZE - 1
lowest_full_row_values = maze_matrix[lowest_full_row, :]

top_zero_cols    = [c for c, val in enumerate(top_row)                if val == 0]
lowest_zero_cols = [c for c, val in enumerate(lowest_full_row_values) if val == 0]

start_coords = None
exit_coords  = None

if top_zero_cols:
    start_col    = (min(top_zero_cols) + max(top_zero_cols)) // 2
    start_coords = (0, start_col)

if lowest_zero_cols:
    exit_col    = (min(lowest_zero_cols) + max(lowest_zero_cols)) // 2
    exit_coords = (lowest_full_row, exit_col)

# ---------------------------------------------------------------------
# 4) Build box center array
# ---------------------------------------------------------------------
def get_center_of_box(br, bc):
    row_start = br * BOX_SIZE
    row_end   = row_start + BOX_SIZE
    col_start = bc * BOX_SIZE
    col_end   = col_start + BOX_SIZE
    center_r  = (row_start + row_end) // 2
    center_c  = (col_start + col_end) // 2
    return (center_r, center_c)

box_centers = []
for br in range(box_rows):
    row_list = []
    for bc in range(box_cols):
        row_list.append(get_center_of_box(br, bc))
    box_centers.append(row_list)

# ---------------------------------------------------------------------
# 5) Map start_coords/end_coords to boxes
# ---------------------------------------------------------------------
def get_box_for_pixel(r, c):
    if r < 0 or r >= box_rows * BOX_SIZE or c < 0 or c >= box_cols * BOX_SIZE:
        return None
    br = r // BOX_SIZE
    bc = c // BOX_SIZE
    return (br, bc)

start_box = get_box_for_pixel(*start_coords) if start_coords else None
end_box   = get_box_for_pixel(*exit_coords)  if exit_coords else None

# ---------------------------------------------------------------------
# 6) Check connection between two centers (horizontal/vertical)
# ---------------------------------------------------------------------
def check_connection(maze_matrix, r1, c1, r2, c2):
    """
    Check if there's a path (no wall) between two centers (r1, c1) and (r2, c2).
    Returns 1 if the path is open (no wall), otherwise 0.
    """
    dr = r2 - r1
    dc = c2 - c1

    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return 1  # same cell, trivial

    step_r = dr / steps
    step_c = dc / steps

    # Horizontal line
    if r1 == r2:
        for step_i in range(steps + 1):
            current_r = r1
            current_c = round(c1 + step_i * step_c)
            if maze_matrix[current_r, current_c] == 1:
                return 0
        return 1

    # Vertical line
    elif c1 == c2:
        for step_i in range(steps + 1):
            current_c = c1
            current_r = round(r1 + step_i * step_r)
            if maze_matrix[current_r, current_c] == 1:
                return 0
        return 1

    return 1

def get_open_closed_paths(maze_matrix, box_centers):
    connections = {}
    for row in range(len(box_centers)):
        for col in range(len(box_centers[row])):
            center_r, center_c = box_centers[row][col]
            paths = {'E': 0, 'W': 0, 'N': 0, 'S': 0}

            # West
            if col - 1 >= 0:
                W_r, W_c = box_centers[row][col - 1]
                paths['W'] = check_connection(maze_matrix, center_r, center_c, W_r, W_c)

            # East
            if col + 1 < len(box_centers[row]):
                E_r, E_c = box_centers[row][col + 1]
                paths['E'] = check_connection(maze_matrix, center_r, center_c, E_r, E_c)

            # North
            if row - 1 >= 0:
                N_r, N_c = box_centers[row - 1][col]
                paths['N'] = check_connection(maze_matrix, center_r, center_c, N_r, N_c)

            # South
            if row + 1 < len(box_centers):
                S_r, S_c = box_centers[row + 1][col]
                paths['S'] = check_connection(maze_matrix, center_r, center_c, S_r, S_c)

            connections[(row, col)] = paths
    return connections

# ---------------------------------------------------------------------
# 7) Build open/closed paths for each cell
# ---------------------------------------------------------------------
open_closed_paths = get_open_closed_paths(maze_matrix, box_centers)

# print("Open/Closed Paths Dictionary:")
# for cell, paths in open_closed_paths.items():
#     print(f"Cell {cell}: {paths}")


# ---------------------------------------------------------------------
# 8) BFS function
# ---------------------------------------------------------------------

def BFS(open_closed_paths, start_box, end_box):
    """
    Returns:
       explored -> list of visited cells in BFS order
       bfsPath  -> dict {child_cell: parent_cell} 
    """
    frontier = [start_box]
    explored = [start_box]
    bfsPath  = {}

    while frontier:
        currCell = frontier.pop(0)
        if currCell == end_box:
            break  # Found the goal

        # Explore E, S, N, W (or any order you want)
        for d in 'ESNW':
            if open_closed_paths[currCell][d] == 1:  # passable
                if d == 'E':
                    childCell = (currCell[0], currCell[1] + 1)
                elif d == 'W':
                    childCell = (currCell[0], currCell[1] - 1)
                elif d == 'N':
                    childCell = (currCell[0] - 1, currCell[1])
                elif d == 'S':
                    childCell = (currCell[0] + 1, currCell[1])

                if childCell not in explored:
                    frontier.append(childCell)
                    explored.append(childCell)
                    bfsPath[childCell] = currCell

    return explored, bfsPath

# ---------------------------------------------------------------------
# 9) Run BFS and reconstruct the path
# ---------------------------------------------------------------------

if start_box is not None and end_box is not None:
    explored, parent_map = BFS(open_closed_paths, start_box, end_box)

    # Reconstruct path from end_box -> start_box
    fwdPath = []
    cell = end_box
    while cell != start_box:
        fwdPath.append(cell)
        cell = parent_map[cell]
    fwdPath.append(start_box)
    fwdPath.reverse()
else:
    explored = []
    fwdPath = []

# ---------------------------------------------------------------------
# 10) Animate the BFS exploration + final path
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(maze_matrix, cmap='gray', origin='upper')
ax.set_title("Maze Solver - BFS Animation")

# Plot all box-centers in red (small)
for br in range(box_rows):
    for bc in range(box_cols):
        (rC, cC) = box_centers[br][bc]
        ax.plot(cC, rC, 'r.', markersize=2)

# Always show Start and End with bigger green markers
if start_box is not None:
    (s_r, s_c) = box_centers[start_box[0]][start_box[1]]
    ax.plot(s_c, s_r, 'bo', markersize=12, label='Start')

if end_box is not None:
    (e_r, e_c) = box_centers[end_box[0]][end_box[1]]
    ax.plot(e_c, e_r, 'bo', markersize=12, label='End')

# We'll animate the BFS in green (explored) & final path in blue
explored_x, explored_y = [], []
path_x, path_y         = [], []

explored_plot, = ax.plot([], [], 'go', markersize=4, label='Explored')
path_plot,     = ax.plot([], [], 'bs', markersize=8, label='Path')

split_index = len(explored)  # The point where BFS is done, path starts

def init():
    explored_plot.set_data([], [])
    path_plot.set_data([], [])
    return explored_plot, path_plot

def animate(i):
    # From 0 to split_index-1, show the BFS exploration
    if i < split_index:
        cell = explored[i]
        (r, c) = box_centers[cell[0]][cell[1]]
        explored_x.append(c)
        explored_y.append(r)
        explored_plot.set_data(explored_x, explored_y)
    else:
        # After BFS is done, start plotting the final path
        j = i - split_index  # so at i=split_index -> fwdPath[0]
        if j < len(fwdPath):
            cell = fwdPath[j]
            (r, c) = box_centers[cell[0]][cell[1]]
            path_x.append(c)
            path_y.append(r)
            path_plot.set_data(path_x, path_y)

    return explored_plot, path_plot

# Total frames = BFS steps + path steps
total_frames = len(explored) + len(fwdPath)
print(fwdPath)

ani = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=total_frames,
    interval=50,   # <<< 50 ms per frame
    blit=True,
    repeat=False
)

plt.legend()
plt.show()

