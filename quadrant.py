def quadrant(pose, map_size=512, block_num=8):
    assert map_size % block_num == 0
    block_size = map_size/block_num
    quad = (int(pose[0]//block_size),int(pose[1]//block_size))
    print(quad)
    return quad

if __name__ == "__main__":
    pose = (191,319)
    quadrant(pose)