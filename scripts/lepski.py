import portion as P


def interval_intersection(interval1, interval2):
    intv1 = P.closed(interval1[0], interval1[1])
    intv2 = P.closed(interval2[0], interval2[1])
    intersection = intv1 & intv2
    if intersection.empty:
        return False
    return True

def lepski_selector(theta_hats, CNFs):
    n = len(theta_hats)
    intervals = [(theta_hats[i] - CNFs[i], theta_hats[i] + CNFs[i]) for i in range(n)]

    # 从最大尺度i往小找
    for i in reversed(range(n)):
        all_intersect = True
        for j in range(i):
            if not interval_intersection(intervals[i], intervals[j]):
                all_intersect = False
                break
        if all_intersect:
            return i
    # 如果一个都不满足，返回最小尺度0
    return 0

# 测试用例
def test_lepski_selector():
    theta_hats = [1.0, 2.0, 1]
    CNFs = [1.0, 0.5, 0.5]

    hat_i = lepski_selector(theta_hats, CNFs)

    print(f"Lepski 选择的索引为: {hat_i}")
    print(f"对应的估计值: {theta_hats[hat_i]}")
    print(f"对应的置信宽度: {CNFs[hat_i]}")

if __name__ == "__main__":
    test_lepski_selector()
