import pickle
import os
import multiprocessing

from models.migrate.exploratory import *

# output folder
OUTPUT_FOLDER = "./models/migrate/output/"


class RunInstance:
    def __init__(self, steps=7500, record=200, **kwargs):
        # instance tissue
        self.tissue = Tissue(**kwargs)
        self.run_folder = self.get_running_environment("cut-random-370")

        # total steps
        self.steps = steps

        # record tallies
        self.record = record

        # surgery period
        self.warm_up, self.surgery_counter = 100, 0
        self.surgery_trigger = 0.99

    @staticmethod
    def get_running_environment(folder):
        # read current run ID
        run_file = open(OUTPUT_FOLDER + "/.run", "r")
        run_id = int(run_file.readline()) + 1

        print("[+] got run ID", run_id)

        if not os.path.exists(OUTPUT_FOLDER + folder):
            os.mkdir(OUTPUT_FOLDER + folder)

        run_folder = OUTPUT_FOLDER + folder + "/run-" + str(run_id) + "/"
        os.mkdir(run_folder)

        # write new run ID
        run_file = open(OUTPUT_FOLDER + "/.run", "w")
        run_file.write(str(run_id))

        return run_folder

    def store_tallies(self):
        pickle.dump(self.tissue.tallies, open(self.run_folder + "/surgery.pkl", "wb"))
        print("[+] store tallies for", self.run_folder)

    @staticmethod
    def get_tally_frame(folder, run_id, output=OUTPUT_FOLDER):
        return pickle.load(open(output + folder + "/run-" + str(run_id) + "/surgery.pkl", "rb"))

    def run(self):
        for step in range(0, self.steps + 1):
            # get cellulatas
            greedy = self.tissue.medium.get_agents(condition=lambda a: isinstance(a, Cellulata) and a.is_greedy())
            releasing = self.tissue.medium.get_agents(condition=lambda a: isinstance(a, Cellulata) and a.is_releasing())
            cellulatas = greedy + releasing

            # break condition
            if not len(cellulatas):
                break

            # collect tallies
            self.tissue.collect_tallies()

            # cell number
            coverage = self.tissue.get_tissue_coverage(cellulatas)

            if step % 50 == 0:
                print("[#] --> coverage   :", coverage, "on ", self.run_folder)

            # record step
            if step % self.record == 0:
                print("[+] simulation", self.run_folder, "time", self.tissue.medium.get_time())

                print("[#] --> greedy   :", len(greedy))
                print("[#] --> releasing :", len(releasing))
                print(f"[#] --> coverage : {coverage:.2f}")

                # store tallies
                self.store_tallies()

            # do surgery warm up
            if coverage > self.surgery_trigger:
                self.surgery_counter += 1
                if step % self.record == self.surgery_trigger // 5:
                    print("[#] --> warm surgery :", self.surgery_counter)

            # perform surgery
            if self.surgery_counter > self.warm_up:
                self.surgery_counter = 0
                self.tissue.perform_surgery()

            # run simulation step
            self.tissue.medium.run()

        # store tallies
        self.store_tallies()


def run_tissue(simulation):
    try:
        simulation.run()
    except AssertionError:
        print("[+] AssertionError on simulation", simulation)


def run():
    print("[+] cellulata migration")

    # run instance
    simulations = [RunInstance(knock_genes={Trait.EPSILON: GENE_CAPACITY, Trait.MOTILITY: (1, 3)})
                   for _ in range(0, 2)]

    # processes
    processes = min(4, len(simulations))

    # simulation pool
    print("[+] spawning", processes, "processes")
    pool = multiprocessing.Pool(processes=processes)
    for each in pool.map(run_tissue, simulations):
        print("[+] run finished")

    # run instance
    simulations = [RunInstance(knock_genes={Trait.MOTILITY: (1, 3)})
                   for _ in range(0, 2)]

    # processes
    processes = min(4, len(simulations))

    # simulation pool
    print("[+] spawning", processes, "processes")
    pool = multiprocessing.Pool(processes=processes)
    for each in pool.map(run_tissue, simulations):
        print("[+] run finished")


# run()

