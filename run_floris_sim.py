from utils import etl
import argparse
import os

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="location of saved OpenFAST simulation data")
    parser.add_argument("--template_dir", type=str, required=True,
                        help="directory containing Turbsim input template, "
                             "InflwWind .dat file, OpenFAST .fst file, and"
                             "FLORIS input .json")
    parser.add_argument("--sim_idx", type=int, required=True,
                        help="simulation_index")
    args = vars(parser.parse_args())

    data_dir = args['data_dir']
    sim_idx = args['sim_idx']
    temp_dir = args['template_dir']

    # Template and input files
    inflow_inp = temp_dir + "NRELOffshrBsline5MW_InflowWind_12mps.dat"
    openfast_inp = temp_dir + "5MW_Land_DLL_WTurb.fst"
    turbsim_inp = temp_dir + "90m_12mps_twr.inp"
    floris_inp = temp_dir + "floris_input.json"

    # For every fifth degree of wind direction, run a FLORIS sim to get Turbsim inputs to farm
    for degree in range(0, 360, 5):
        sim_dir = data_dir + "sim_{0:03d}_deg_{1:03d}/".format(sim_idx, degree)
        os.mkdir(sim_dir)

        etl.generate_turbsim_input(destination=sim_dir,
                                   turbsim_template=turbsim_inp,
                                   inflow_baseline=inflow_inp,
                                   openfast_fst=openfast_inp,
                                   config='farm',
                                   farm_json=floris_inp,
                                   wind_direction=float(degree))




