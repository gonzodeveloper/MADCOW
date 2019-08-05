from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import argparse
import os
import etl


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="location to save simulation runs")
    parser.add_argument("--num_simulations", type=int, required=True,
                        help="number of simulations to run")
    parser.add_argument("--wind_speed", type=float, required=True,
                        help="windspeed for simulations")
    parser.add_argument("--iec_class", type=str, required=True,
                        help="IEC turbulence characeteristic (A, B, or C)")
    parser.add_argument("--template_dir", type=str, required=True,
                        help="location of input templates")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--single", action="store_true",
                       help="simulations will be run on a single turbine")
    group.add_argument("--farm", action="store_true",
                       help="simulations will be run on a farm (one sim will run all incoming wind directions to farm")

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    # Make simulation directories in data_dir
    simulation_dirs = [args['data_dir'] + "sim_{0:04d}/".format(sims)
                       for sims in range(args['num_simulations'])]
    for dirs in simulation_dirs:
        os.mkdir(dirs)

    # Get template_files
    turbsim_temp = args['template_dir'] + "90m_12mps_twr.inp"
    inflow_baseline = args['template_dir'] + "NRELOffshrBsline5MW_InflowWind_12mps.dat"
    openfast_fst = args['template_dir'] + "5MW_Land_DLL_WTurb.fst"

    with Pool(processes=cpu_count()) as pool:
        results = [pool.apply_async(etl.run_openfast_sim, args=(dirs, args['wind_speed'], args['iec_class'],
                                                                turbsim_temp, inflow_baseline, openfast_fst))
                   for dirs in simulation_dirs]

        for idx, r in enumerate(results):
            etl.print_progress_bar(idx, len(results))
            r.get()

