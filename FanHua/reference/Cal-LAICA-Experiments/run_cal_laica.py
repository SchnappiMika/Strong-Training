from time import time

from cal_laica.runner import Runner
from cal_laica.utils.Config import Config
from cal_laica.utils.Parser import Parser


def main(train=True, inc=-1, hyper=-1, base=-1):
    t = time()
    args = Parser().get_parser().parse_args()

    if inc >= 0 and hyper >= 0 and base >= 0:
        args.inc = inc
        args.hyper = hyper
        args.base = base

    config = Config(args)

    solver = Runner(config=config)

    # solver.random_eval(episodes=1000)
    # Training mode
    if train:
        solver.run()
        # solver.online_run()

    # # Evaluation mode
    # solver.eval(10)

    print(time() - t)


if __name__ == '__main__':
    main(train=True, inc=-1, hyper=-1)
