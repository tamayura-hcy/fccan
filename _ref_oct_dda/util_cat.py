import torch

def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def unsorted_segment_sum_device(data, segment_ids, num_segments,device):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).to(device).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


if __name__ == '__main__':
    print('validate the pytorch implementation')

    import tensorflow as tf
    c = tf.constant([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    result2 = tf.unsorted_segment_sum(c, tf.constant([2, 1, 1]), 3)
    result4 = tf.unsorted_segment_sum(c, tf.constant([2, 0, 1]), 3)
    result5 = tf.unsorted_segment_sum(c, tf.constant([3, 1, 0]), 5)
    sess = tf.Session()
    print("result2")
    print(sess.run(result2))
    print("result4")
    print(sess.run(result4))
    print("result5")
    print(sess.run(result5))

    d = torch.tensor([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
    print(unsorted_segment_sum(d, torch.tensor([2, 1, 1]), 3))
    print(unsorted_segment_sum(d, torch.tensor([2, 0, 1]), 3))
    print(unsorted_segment_sum(d, torch.tensor([3, 1, 0]), 5))
