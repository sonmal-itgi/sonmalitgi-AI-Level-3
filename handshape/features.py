"""2D 관절 각도 15개. 손목 위치·손 크기·좌우에 불변, 카메라 투영에는 불변하지 않음."""
import math
SCHEMA = 'handshape-angles2d-v1'
TRIPLES = [(a, a+1, a+2) for a in (1, 5, 9, 13, 17)] + [(a+1, a+2, a+3) for a in (1, 5, 9, 13, 17)] + [(0, a, a+1) for a in (1, 5, 9, 13, 17)]
def features(points, width, height):
    if len(points) != 21 or width <= 0 or height <= 0:
        raise ValueError('완전한 손 좌표와 영상 크기가 필요합니다')
    xy = [(p['x'] * width, p['y'] * height) for p in points]
    if not all(math.isfinite(v) for p in xy for v in p):
        raise ValueError('유효하지 않은 좌표')
    result = []
    for a,b,c in TRIPLES:
        u = (xy[a][0]-xy[b][0], xy[a][1]-xy[b][1])
        v = (xy[c][0]-xy[b][0], xy[c][1]-xy[b][1])
        norm = math.hypot(*u)*math.hypot(*v)
        if norm < 1e-8:
            raise ValueError('겹친 관절 좌표')
        result.append(math.acos(max(-1, min(1, (u[0]*v[0]+u[1]*v[1])/norm)))/math.pi)
    return result
