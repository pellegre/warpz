from models.colony.biofilm.species import *

import pandas

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.animation


# ===========
#
# plots
#
# ===========

# grid size
GRID = (6, 6)


# upper bounds
TOP = 1.00
LENGTH = 1.00

# energy seed
WARPS_SEED = int(CHAMBER_MOLES)


# ===========
#
# bio film
#
# ===========

class SplashSpecie(Mother):
    def __init__(self, **kwargs):
        # anabolic target molecule
        target = "CNCNCNCN"

        # reaction units
        anabolic = [("C", "N"), ("CN", "CN"), ("CNCN", "CN"), ("CNCNCN", "CN"), ("CNCN", "CNCN")]
        catabolic = [("C", "C"), ("C", "CC"), ("CC", "CC"), ("C", "CCC"), ("CC", "CC")]

        # pump space
        pressure = ["C", "N", "CN", "CC", "CNCN", "CNCNCN", "CNCNCNCN",  "CCC", "CCCC"]

        super().__init__(target=target, anabolic=anabolic, catabolic=catabolic,
                         pressure=pressure, matrix=False, **kwargs)

        # set moles
        self._anabolic_drive.set_moles(1)
        self._catabolic_drive.set_moles(1)
        self._pressure_drive.set_moles(1)
        self._pleasure_drive.set_moles(1)


class Splash(Specie):
    def __init__(self, universe, position, replications=0):
        # setup gene
        chromosome = SplashSpecie()

        # initial container
        container = {"CNCNCNCN": CHAMBER_MOLES // 4, "CNCNCN": CHAMBER_MOLES // 4, "CNCN": CHAMBER_MOLES // 4,
                     "CN": CHAMBER_MOLES // 4, "Z": CHAMBER_MOLES // 8}

        # initialize
        super().__init__(universe=universe, position=position, mother=chromosome, container=container)

        # done
        self.done = False

        # replications
        self.replications = replications

    def splash(self):
        # get anabolic drive moles
        moles = self.get_substrate(self._chromosome.get_anabolic_drive())

        if moles > CHAMBER_MOLES // 2:
            # get substrates
            substrates = self.get_container()

            # craft container for new born
            born_container = dict()
            for s in substrates:
                # splash it
                born_container[s] = substrates[s] // 2

                # balance
                self.put_substrate(s, -born_container[s])

            # copy chromosome
            self._chromosome.mutate()

            # put back container
            self.get_cell().put_substrates(born_container)

            # dissipate reward
            reward = self.get_cell().get_substrate("R")
            self.get_cell().put_substrate("R", -reward)

            # count replication
            self.replications += 1

    # ======
    # interact
    #
    def interact(self, signals):
        if not self.done:
            # diffuse
            self.diffuse_inward()

            # sense the environment
            self.sense()

            # catalytic reaction
            self.react_and_pump()

            # thermal reactions
            self.thermal()

            # diffuse
            self.diffuse_outward()

            # move
            yield self.move()

            # propagate
            self.propagate()

            # coco splash
            yield self.splash()

        else:
            # gone
            yield Gone()


class BioFilm:
    def __init__(self):
        # display cycle
        self.display_cycle = 40

        # medium
        self.medium = Dish(upper_bounds=[LENGTH, TOP], bottom_bounds=[0.0, 0.0], boundary=Box.Boundary.CLOSED,
                           tracking_type=Box.Tracking.QUAD, chemical=FilmChemical(warps=WARPS_SEED), grid=GRID)

        for cell in self.medium.get_cells():
            cell.put_substrate("CCCC", WARPS_SEED)
            cell.put_substrate("CCC", WARPS_SEED)
            cell.put_substrate("CC", WARPS_SEED)

        # coconata
        self.medium.put(Splash(universe=self.medium, position=[0.50, 0.50]))

        # time counter
        self.tic = time.process_time()

        # tallies
        self.pump_stats, self.reaction_stats = dict(), dict()

    @staticmethod
    def is_c_molecule(substrate):
        return all([c == 'C' for c in substrate])

    @staticmethod
    def is_cn_molecule(substrate):
        return len(substrate) % 2 == 0 and substrate.count("CN") == len(substrate) // 2

    def step(self):
        # run
        self.medium.run()

        # get substrates
        cell_substrates = self.medium.get_substrates(condition=lambda a: isinstance(a, Block))

        # cellulatas
        coconatas = list(self.medium.get_agents(condition=lambda a: isinstance(a, Particle)))

        # accumulate cell tallies
        for coco in coconatas:
            # accumulate state
            for pump in coco.get_chromosome().get_pumps():
                decision = pump.get_action()

                # pump
                each = pump.get_name()[pump.get_name().find("@") + 1:]

                if decision is not Action.NONE:
                    if each not in self.pump_stats:
                        self.pump_stats[each] = {Action.INWARD: 0, Action.OUTWARD: 0, Action.BUILD: 0, Action.BREAK: 0}

                    self.pump_stats[each][decision] += 1

            for reaction in coco.get_chromosome().get_reactions():
                decision = reaction.get_action()
                each = reaction.get_name()[reaction.get_name().find("@") + 1:]

                if decision is not Action.NONE:
                    if each not in self.reaction_stats:
                        self.reaction_stats[each] = {Action.POLYMERIZATION: 0, Action.CLEAVAGE: 0}

                    self.reaction_stats[each][decision] += 1

        # print some stats
        if self.medium.get_time() % self.display_cycle == 0 or self.medium.get_time() == 1:
            # check time
            toc = time.process_time()

            # chemokines (in unit cells)
            cells = self.medium.get_agents(condition=lambda a: isinstance(a, Cell))

            # get token
            warp = self.medium.get_chemistry().get_token()

            # warps in cell
            cell_warps = sum([cell.get_substrate(warp) for cell in cells])

            # warps in the dish
            dish_warps = self.medium.get_substrate(warp)

            # coconatas warps
            coconata_warps = sum([coconata.get_substrate(warp) for coconata in
                                  self.medium.get_agents(condition=lambda a: isinstance(a, Coconata))])

            # total warps
            total_warps = dish_warps + cell_warps + coconata_warps

            print("-------------- (step) ", self.medium.get_time())
            print(f"[-] time             : {toc - self.tic:.2f}")

            print("[-] coconatas        :", len(coconatas))

            print(f"[#] --> warps  ({total_warps:.2f})")
            print(f"[#] dish             : {dish_warps:.2f}")
            print(f"[#] cells            : {cell_warps:.2f}")
            print(f"[#] coconatas        : {coconata_warps:.2f}")

            # dissipated energy
            dissipated_energy = self.medium.get_dissipated_energy()

            # cell energy (bonds and free)
            cell_free_energy = self.medium.get_free_energy(condition=lambda a: isinstance(a, Block))
            cell_bond_energy = self.medium.get_bond_energy(condition=lambda a: isinstance(a, Block))

            # coconata energy (bonds and free)
            coconata_free_energy = self.medium.get_free_energy(condition=lambda a: isinstance(a, Coconata))
            coconata_bond_energy = self.medium.get_bond_energy(condition=lambda a: isinstance(a, Coconata))

            # total energy
            total_energy = cell_free_energy + cell_bond_energy + coconata_bond_energy + \
                           coconata_free_energy + dissipated_energy

            # print stats
            print(f"[#] --> energy ({total_energy:.2f})")
            print(f"[#] dissipated       : {dissipated_energy:.2f}")
            print(f"[#] bond (cell)      : {cell_bond_energy:.2f}")
            print(f"[#] free (cell)      : {cell_free_energy:.2f}")
            print(f"[#] bond (coconata)  : {coconata_bond_energy:.2f}")
            print(f"[#] free (coconata)  : {coconata_free_energy:.2f}")

            moles = [c.get_moles() for c in coconatas]
            print(f"[#] average moles    : {sum(moles) / len(moles):.2f}")

            cell_elements = self.medium.get_elements(condition=lambda a: isinstance(a, Block))

            print("[#] --> substrates (cell)   ", {s: cell_substrates[s] for i, s in enumerate(sorted(
                cell_substrates, key=lambda r: cell_substrates[r], reverse=True))})

            print("[#] --> elements   (cell)   ", cell_elements)

            coconata_substrates = self.medium.get_substrates(condition=lambda a: isinstance(a, Coconata))
            coconata_elements = self.medium.get_elements(condition=lambda a: isinstance(a, Coconata))

            print("[#] --> substrates (coco)   ", {s: coconata_substrates[s] for i, s in enumerate(sorted(
                coconata_substrates, key=lambda r: coconata_substrates[r], reverse=True))})

            print("[#] --> elements   (coco)   ", coconata_elements)
            print("[#] --> elements   (total)  ", {e: coconata_elements[e] + cell_elements[e]
                                                   for e in coconata_elements})

            for i, each in enumerate(coconatas):
                print("[#] unbalanced       :", "true" if each.is_unbalanced() else "false")
                print("[#] starved          :", "true" if each.is_starved() else "false")
                print("[#] scarce           :", "true" if each.is_scarce() else "false")
                print("[#] replications     :", each.replications)

            # get frames
            if len(self.reaction_stats) > 0:
                reactions = pandas.DataFrame(self.reaction_stats)
                reactions.index = ["POLY", "BREAK"]

                print("[+] reactions")
                print(reactions)

            if len(self.pump_stats) > 0:
                pumps = pandas.DataFrame(self.pump_stats)
                pumps.index = ["INWARD", "OUTWARD", "BUILD", "BREAK"]

                # print pumps
                print("[+] pumps")
                print(pumps)

            # update time
            self.tic = toc


biofilm = BioFilm()

# figure
fig = plt.figure()

ax_pump = fig.add_subplot(211)
ax_reaction = fig.add_subplot(212)


def animate(i):
    # step
    biofilm.step()

    # cellulatas
    coco = list(biofilm.medium.get_agents(condition=lambda a: isinstance(a, Particle)))[0]

    # get frames
    reactions = pandas.DataFrame({s[0] + s[1]: coco.reaction_stats[s] for s in coco.reaction_stats})
    pumps = pandas.DataFrame(coco.pump_stats)

    ax_pump.clear()
    ax_reaction.clear()

    ax_pump.set_title("pump")
    ax_reaction.set_title("reaction")

    pumps.transpose().plot.bar(rot=0, ax=ax_pump)
    reactions.transpose().plot.bar(rot=0, ax=ax_reaction)


def animation():
    print("[+] proteo benchmark")

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=25, repeat=True)

    plt.show()


def main():
    print("[+] proteo benchmark")

    while True:
        biofilm.step()


main()
