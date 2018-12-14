import abc
import numpy as np
from tfutils.pyutils import inheritdocstring


class WeightInitializer(abc.ABC):
    """
    Args:
        input_dim: Positive integer.
        output_dim: Positive integer.
    """

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abc.abstractmethod
    def initialize(self):
        """Returns a np.array with shape `(self.input_dim, self.output_dim)`
        as the initialized weight."""
        pass


@inheritdocstring
class XavierInitializer(WeightInitializer):
    """
    Args:
        distribution: String. Either "normal" or "uniform".
    """

    def __init__(self, distribution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distribution = distribution

    def initialize(self):
        scale = np.sqrt(2 / (self.input_dim + self.output_dim))
        size = (self.input_dim, self.output_dim)

        if self.distribution == 'uniform':
            return np.random.uniform(low=-scale, high=scale, size=size)

        elif self.distribution == 'normal':
            return np.random.normal(loc=0.0, scale=scale, size=size)

        else:
            raise ValueError()


def simulate_output_comp_stds(kernal_initializer,
                              batch_size,
                              input_dim,
                              output_dim):
    """Simulates the standard derivatives for each component of the output-
    batch

        output_batch = np.matmul(input_batch, weight)

    where the `weight` is initialized by the `kernel_initializer`.

    Args:
        kernel_initializer: Instance inherting `WeightInitializer`.
        batch_size: Positive integer.
        input_dim: Positive integer.
        output_dim: Positive integer.

    Returns:
        Numpy array with shape `(output_dim)`.
    """
    input_batch = np.random.random(size=(batch_size, input_dim))
    weight = kernal_initializer.initialize()
    output_batch = np.matmul(input_batch, weight)
    output_comp_stds = []  # standard derivative for each component of output.
    for comp in range(output_dim):
        output_comp_batch = output_batch[:, comp]
        output_comp_stds.append(np.std(output_comp_batch))
    return output_comp_stds


def main(args):
    if args.kernel_initializer == 'xavier':
        kernel_initializer = XavierInitializer(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            distribution=args.distribution)
    else:
        raise ValueError()

    for _ in range(args.simulate_times):
        output_comp_stds = simulate_output_comp_stds(
            kernel_initializer, args.batch_size,
            args.input_dim, args.output_dim)
        print('Mean value of standard derivatives of output components:',
              '{0:.3f} ({1:.3f})'.format(np.mean(output_comp_stds),
                                         np.std(output_comp_stds)))


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--simulate_times', type=int, default=1,
                        help='Times of simulation')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--output_dim', type=int)
    parser.add_argument('--kernel_initializer', type=str, default='xavier')
    parser.add_argument('--distribution', type=str, default='uniform')
    args = parser.parse_args()

    main(args)
