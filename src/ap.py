"""Calculate AverP."""
import sets


def calculate_ap(labels, logits, class_num=20):
    """Calculate AverP.

    Args:
        labels: groundtruth list, e.g: ['001', 0, 0, 1 ...(*20)]
        logits: predicted list, e.g: ['001', 0.2, 0.1, 0.9 ...(*20)]
    """
    sum_p = 0.0
    for cur_class_id in xrange(class_num):
        sum_p += calculate_ap_for_class(labels, logits, cur_class_id)
    return sum_p / float(class_num)


def calculate_ap_for_class(labels, logits, cur):
    """Calculate AverP for a certain class."""
    cur += 1
    sorted_logits = sorted(logits, key=lambda x: x[cur], reverse=True)
    gt_set = sets.Set([])
    for item in labels:  # accumulate correct images
        if item[cur] == 1:
            gt_set.add(item[0])
    corr_total, total, cur_sp = 0, 0, 0.0
    for item in sorted_logits:
        total += 1
        if item[0] in gt_set:
            corr_total += 1
            cur_sp += float(corr_total) / float(total)
    return cur_sp / float(corr_total)


def main():
    """Test Module."""
    gd_list = [
        ['001.jpg', 1, 0, 0],
        ['002.jpg', 0, 1, 0],
        ['003.jpg', 0, 0, 1],
        ['004.jpg', 1, 0, 0],
        ['005.jpg', 1, 0, 0],
    ]
    test_list = [
        ['001.jpg', 0.5, 0.2, 0.3],
        ['002.jpg', 0.6, 0.2, 0.3],
        ['003.jpg', 0.3, 0.2, 0.3],
        ['004.jpg', 0.2, 0.2, 0.3],
        ['005.jpg', 0.9, 0.2, 0.3],
    ]
    print(calculate_ap_for_class(gd_list, test_list, 0))
    print(calculate_ap(gd_list, test_list, 3))


if __name__ == '__main__':
    main()
