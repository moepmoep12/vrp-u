import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from vrpu.solver import SolvingSnapshot


class ISolverRenderer(ABC):
    """
    Interface. Renders solving process of a solver
    """

    @abstractmethod
    def render_solver_history(self, history: [SolvingSnapshot], **kwargs) -> None:
        """
        Renders solving process of a solver.
        :param history: The collection of snapshots collected during solving
        :param kwargs: Key worded arguments
        """
        pass


class SolverRenderer(ISolverRenderer):

    def render_solver_history(self, history: [SolvingSnapshot], **kwargs) -> None:

        full_screen: bool = kwargs.get('full_screen', True)

        best_values = [h.best_value for h in history]
        mean_values = [h.average for h in history]
        steps = [h.step for h in history]
        axis_names = ['Best', 'Mean']
        values = [best_values, mean_values]

        fig, a = plt.subplots(1, 1, figsize=(20, 9))
        for i in range(len(axis_names)):
            a.plot(steps, values[i], label=axis_names[i])

        a.set_title('Solver History')
        a.grid(True)
        a.set_xlabel("Step")
        a.set_ylabel("Total distance")
        a.legend()

        # windowed full screen
        if full_screen:
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed')

        plt.show()
