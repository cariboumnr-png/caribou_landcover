# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Utility script to generate dummy/mock geospatial datasets (GeoTIFFs)
and corresponding configurations for local pipeline runs and testing.
'''

import os
import sys
import argparse
import landseg.testing as testing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        nargs='?',
        default='./experiment/input',
        help='Directory where dummy data will be generated.'
    )
    parser.add_argument(
        '-y',
        '--yes',
        action='store_true',
        help='Automatically confirm overwriting existing files.'
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f'WARNING: Target directory "{args.output_dir}" '
            f'already exists and is not empty.'
        )
        if args.yes:
            print(' -> Overwrite existing files as configured')
        else:
            response = input(
                ' -> Generating dummy data will overwrite existing files. '
                'Proceed? [y/N]: '
            )
            if response.strip().lower() not in ('y', 'yes'):
                print('Aborted.')
                sys.exit(0)

    print('-' * 10)
    testing.generate_dummy_data(args.output_dir)
