import numpy as np
from collections import deque 

def bfs(obstacle, explored, frontier, start):

    height, width = obstacle.shape
    queue = deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if frontier[x][y] == 1:
            #print(len(path))
            return len(path)
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and obstacle[x2][y2] == 0 and explored[x2][y2] == 1 and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    return 0
      
