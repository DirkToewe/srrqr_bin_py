# This file is part of QuIPPy.
#
# QuIPPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# QuIPPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with QuIPPy. If not, see <http://www.gnu.org/licenses/>.
'''
Created on Nov 8, 2019

@author: Dirk Toewe
'''

from distutils.core import setup

setup(
  name='srrqr',
  version='0.0.0',
  description='A proof-of-concept implementation of an improved strong rank-revealing QR decomposition.',
  author='Dirk Toewe',
  url='https://github.com/DirkToewe/srrqr_bin_py',
  packages=(
    'srrqr'
  )
)
