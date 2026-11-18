"""Choose checkpointed transformer layers without changing model weights."""


def checkpointed_indices(n_layer: int, every_n: int | None = None) -> tuple[int, ...]:
    """Return zero-based block indices checkpointed every Nth layer.

    None selects no layers. A value of 1 selects all layers; 2 selects
    indices 1, 3, 5, ... (counting from layer 1). This is a *policy*,
    not a statement about measured memory savings.
    """
    if type(n_layer) is not int or n_layer < 1:
        raise ValueError("n_layer must be a positive integer")
    if every_n is None:
        return ()
    if type(every_n) is not int or not 1 <= every_n <= n_layer:
        raise ValueError("every_n must be an integer from 1 through n_layer")
    return tuple(index for index in range(n_layer) if (index + 1) % every_n == 0)
