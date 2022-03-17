import json
import os
import re
import logging
import logging.config
from tkinter import Tk, ttk, RAISED, Label, IntVar, Entry, W, DoubleVar, BooleanVar, Checkbutton, Button, filedialog, \
    X, StringVar, SUNKEN

import yaml

from vrpu.core import UTurnTransitionFunction, UTurnGraph
from vrpu.core.graph.graph import Graph
from vrpu.core.graph.search import NodeDistanceDijkstra
from vrpu.core.route_planning.serializable import JSONSerializer
from vrpu.util import RandomGridGraphGenerator, RandomTransportRequestGenerator
from vrpu.visualization import GraphRenderer

logger = logging.getLogger('simulator')


class ProblemCreatorGUI:
    def __init__(self):
        self.created_graph = None
        self.chosen_graph = None
        self.chosen_graph_path = ''
        self.created_trqs = []

    def create_gui(self, root):
        tab_control = ttk.Notebook(root)
        tab_graph = ttk.Frame(tab_control)
        tab_scenario = ttk.Frame(tab_control)
        tab_control.add(tab_scenario, text='Create scenario')
        tab_control.add(tab_graph, text='Create graph')
        tab_control.pack(expand=1, fill='both')

        self._create_problem_frame(tab_scenario)
        self._create_graph_frame(tab_graph)

    def _create_problem_frame(self, root):
        settings_frame = ttk.Frame(master=root, relief=RAISED, borderwidth=1)

        # trq count
        lbl_trq_count = Label(master=settings_frame, text='Transport Requests')
        lbl_trq_count.grid(column=0, row=0, sticky=W, padx=5, pady=5)
        self.trq_count = IntVar()
        self.trq_count.set(10)
        entry_trq_count = Entry(master=settings_frame, textvariable=self.trq_count)
        entry_trq_count.grid(column=1, row=0, sticky=W, padx=5, pady=5)

        # vehicle count
        lbl_vehicle_count = Label(master=settings_frame, text='Vehicles')
        lbl_vehicle_count.grid(column=0, row=1, sticky=W, padx=5, pady=5)
        self.vehicle_count = IntVar()
        self.vehicle_count.set(4)
        entry_vehicle_count = Entry(master=settings_frame, textvariable=self.vehicle_count)
        entry_vehicle_count.grid(column=1, row=1, sticky=W, padx=5, pady=5)

        # max capacity
        lbl_max_capacity = Label(master=settings_frame, text='Max Capacity')
        lbl_max_capacity.grid(column=0, row=2, sticky=W, padx=5, pady=5)
        self.max_capacity = IntVar()
        self.max_capacity.set(10)
        entry_max_capacity = Entry(master=settings_frame, textvariable=self.max_capacity)
        entry_max_capacity.grid(column=1, row=2, sticky=W, padx=5, pady=5)

        # depot
        lbl_depot = Label(master=settings_frame, text='Depot')
        lbl_depot.grid(column=0, row=3, sticky=W, padx=5, pady=5)
        self.depot = StringVar()
        entry_depot = Entry(master=settings_frame, textvariable=self.depot)
        entry_depot.grid(column=1, row=3, sticky=W, padx=5, pady=5)

        # pickup & delivery
        self.pd = BooleanVar()
        self.pd.set(False)
        check_pd = Checkbutton(settings_frame, text="PickUp & Delivery?",
                               variable=self.pd)
        check_pd.grid(column=0, row=4, sticky=W, padx=5, pady=5)

        settings_frame.pack(fill=X)

        # graph frame
        graph_frame = ttk.Frame(master=root, relief=RAISED, borderwidth=1)
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.columnconfigure(1, weight=3)
        graph_frame.columnconfigure(2, weight=1)

        label_path = Label(master=graph_frame, text="Graph")
        label_path.grid(column=0, row=0, sticky=W, padx=10)

        label_path_chosen = Label(master=graph_frame, text="....", relief=SUNKEN, width=45)
        label_path_chosen.grid(column=1, row=0, sticky=W)

        def on_choose_graph_clicked():
            filename = filedialog.askopenfilename(defaultextension='.json', filetypes=[("json files", '*.json')])
            if not filename:
                return
            label_path_chosen['text'] = filename
            self.chosen_graph_path = filename
            with open(filename) as f:
                self.chosen_graph = Graph.from_json(json.load(f))

        btn_choose = Button(master=graph_frame, text="Choose graph", relief=RAISED,
                            command=on_choose_graph_clicked)
        btn_choose.grid(column=2, row=0, padx=10, sticky=W)

        def on_show_graph():
            if not self.chosen_graph:
                return
            GraphRenderer().render_graph(self.chosen_graph)

        btn_show_graph = Button(master=graph_frame, text="Show graph", relief=RAISED,
                                command=on_show_graph)
        btn_show_graph.grid(column=2, row=1, padx=10, sticky=W)

        graph_frame.pack(fill=X)

        # create frame
        create_frame = ttk.Frame(master=root, relief=RAISED, borderwidth=1)

        create_btn = Button(create_frame, text='Create Scenario', command=self._create_scenario, height=4, width=20)
        create_btn.grid(column=1, row=0, sticky=W, padx=10)

        def save_scenario():

            if not self.chosen_graph:
                return

            if not self.created_trqs or len(self.created_trqs) == 0:
                return

            directory = filedialog.askdirectory()
            if not directory:
                return

            pattern = re.compile(r"\d+x\d+")
            result = pattern.search(self.chosen_graph_path)
            graph_name = result.group(0)

            with open(self.chosen_graph_path) as f:
                graph_data = json.load(f)

            scenario_path = f"{directory}/" \
                            f"{'vrpdp' if self.pd.get() else 'cvrp'}_{graph_name}_{self.trq_count.get()}.json"

            result = {
                'depot': self.depot.get(),
                'vehicle_count': self.vehicle_count.get(),
                'max_capacity': self.max_capacity.get(),
                'graph': graph_data,
                'requests': self.created_trqs
            }

            with open(scenario_path, 'w') as file:
                file.write(json.dumps(result, indent=4, cls=JSONSerializer))
                logger.info(f"Saved scenario {scenario_path}")

        save_btn = Button(create_frame, text='Save scenario', command=save_scenario, height=4, width=20)
        save_btn.grid(column=2, row=0, sticky=W, padx=10)

        create_frame.pack(fill=X)

    def _create_graph_frame(self, root):
        settings_frame = ttk.Frame(master=root, relief=RAISED, borderwidth=1)

        # size x
        lbl_size_x = Label(master=settings_frame, text='Size X')
        lbl_size_x.grid(column=0, row=0, sticky=W, padx=5, pady=5)
        self.size_x = IntVar()
        self.size_x.set(10)
        entry_size_x = Entry(master=settings_frame, textvariable=self.size_x)
        entry_size_x.grid(column=1, row=0, sticky=W, padx=5, pady=5)

        # size y
        lbl_size_y = Label(master=settings_frame, text='Size Y')
        lbl_size_y.grid(column=0, row=1, sticky=W, padx=5, pady=5)
        self.size_y = IntVar()
        self.size_y.set(10)
        entry_size_y = Entry(master=settings_frame, textvariable=self.size_y)
        entry_size_y.grid(column=1, row=1, sticky=W, padx=5, pady=5)

        # increments range
        lbl_range = Label(master=settings_frame, text='Distance increment range')
        lbl_range.grid(column=0, row=2, sticky=W, padx=5, pady=5)
        self.range_from = IntVar()
        self.range_from.set(100)
        self.range_to = IntVar()
        self.range_to.set(100)
        entry_range_from = Entry(master=settings_frame, textvariable=self.range_from)
        entry_range_from.grid(column=1, row=2, sticky=W, padx=5, pady=5)
        entry_range_to = Entry(master=settings_frame, textvariable=self.range_to)
        entry_range_to.grid(column=2, row=2, sticky=W, padx=5, pady=5)

        # drop node probability
        lbl_drop_node = Label(master=settings_frame, text='Drop node probability')
        lbl_drop_node.grid(column=0, row=3, sticky=W, padx=5, pady=5)
        self.drop_node = DoubleVar()
        self.drop_node.set(0.10)
        entry_drop_node = Entry(master=settings_frame, textvariable=self.drop_node)
        entry_drop_node.grid(column=1, row=3, sticky=W, padx=5, pady=5)

        # drop edge probability
        lbl_drop_edge = Label(master=settings_frame, text='Drop edge probability')
        lbl_drop_edge.grid(column=0, row=4, sticky=W, padx=5, pady=5)
        self.drop_edge = DoubleVar()
        self.drop_edge.set(0.07)
        entry_drop_edge = Entry(master=settings_frame, textvariable=self.drop_edge)
        entry_drop_edge.grid(column=1, row=4, sticky=W, padx=5, pady=5)

        # remove unreachable nodes
        self.remove_unreachable = BooleanVar()
        self.remove_unreachable.set(True)
        check_remove_unreachable = Checkbutton(settings_frame, text="Remove unreachable nodes",
                                               variable=self.remove_unreachable)
        check_remove_unreachable.grid(column=0, row=5, sticky=W, padx=5, pady=5)

        settings_frame.pack(fill=X)

        # graph frame
        graph_frame = ttk.Frame(master=root, relief=RAISED, borderwidth=1)
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.columnconfigure(1, weight=3)
        graph_frame.columnconfigure(2, weight=1)

        label_path = Label(master=graph_frame, text="Graph")
        label_path.grid(column=0, row=0, sticky=W, padx=10)

        label_path_chosen = Label(master=graph_frame, text="....", relief=SUNKEN, width=45)
        label_path_chosen.grid(column=1, row=0, sticky=W)

        def on_choose_graph_clicked():
            filename = filedialog.askopenfilename(defaultextension='.json', filetypes=[("json files", '*.json')])
            if not filename:
                return
            label_path_chosen['text'] = filename
            self.chosen_graph_path = filename
            with open(filename) as f:
                self.created_graph = Graph.from_json(json.load(f))

        btn_choose = Button(master=graph_frame, text="Choose graph", relief=RAISED,
                            command=on_choose_graph_clicked)
        btn_choose.grid(column=2, row=0, padx=10, sticky=W)

        graph_frame.pack(fill=X)

        # create frame
        run_frame = ttk.Frame(master=root, relief=RAISED, borderwidth=1)

        create_btn = Button(run_frame, text='Create Graph', command=self._create_graph, height=4, width=20)
        create_btn.grid(column=0, row=0, sticky=W, padx=10)

        def on_show_graph():
            if not self.created_graph:
                return
            GraphRenderer().render_graph(self.created_graph)

        btn_show_graph = Button(master=run_frame, text="Show graph",
                                command=on_show_graph, height=4, width=20)
        btn_show_graph.grid(column=1, row=0, padx=10, sticky=W)

        def on_calc_distances():
            if not self.created_graph:
                return

            node_dist_fct = NodeDistanceDijkstra(dict())
            logging.info('Calculating normal node distances...')
            node_dist_fct.calculate_distances(self.created_graph)

            u_graph = UTurnGraph(UTurnTransitionFunction(self.created_graph), self.created_graph)
            node_dist_fct.calculate_distances(u_graph)

            setattr(self.created_graph, '_distances', node_dist_fct.distance_dict)

        btn_calc_distances = Button(master=run_frame, text="Calculate Distances",
                                    command=on_calc_distances, height=4, width=20)
        btn_calc_distances.grid(column=1, row=0, padx=10, sticky=W)

        def save_graph():

            if not self.created_graph:
                return

            if not self.chosen_graph_path:
                directory = filedialog.askdirectory()
                if not directory:
                    return

                graph_name = f"{self.size_x.get()}x{self.size_y.get()}"
                graph_path = f"{directory}/{graph_name}.json"
            else:
                graph_path = self.chosen_graph_path

            with open(graph_path, 'w') as file:
                serialized = json.dumps(self.created_graph, indent=4, cls=JSONSerializer)
                file.write(serialized)
                logger.info(f"Saved graph {graph_path}")

        save_btn = Button(run_frame, text='Save Graph', command=save_graph, height=4, width=20)
        save_btn.grid(column=2, row=0, sticky=W, padx=10)

        run_frame.pack(fill=X)

    def _create_graph(self):
        generator = RandomGridGraphGenerator()
        self.created_graph = generator.generate_graph(size_x=self.size_x.get(),
                                                      size_y=self.size_y.get(),
                                                      increments_between_nodes=list(
                                                          range(self.range_from.get(), self.range_to.get() + 1)),
                                                      drop_edge_prob=self.drop_edge.get(),
                                                      drop_node_prob=self.drop_node.get(),
                                                      remove_unreachable_u_nodes=self.remove_unreachable.get())

        renderer = GraphRenderer()
        renderer.render_graph(self.created_graph)

    def _create_scenario(self):
        if not self.chosen_graph:
            return

        self.created_trqs = RandomTransportRequestGenerator().generate_transport_requests(self.chosen_graph,
                                                                                          self.trq_count.get(),
                                                                                          self.depot.get(),
                                                                                          [self.depot.get()],
                                                                                          pick_up=self.pd.get())

        GraphRenderer().render_graph(self.chosen_graph, trqs=self.created_trqs, depot=self.depot.get())


if __name__ == '__main__':
    root = Tk(className='Problem Creator')
    root.geometry("600x400")

    with open(os.path.join(os.path.dirname(__file__), 'logging.yaml')) as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

    simulator_gui = ProblemCreatorGUI()
    simulator_gui.create_gui(root)

    root.mainloop()
