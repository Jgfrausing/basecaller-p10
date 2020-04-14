# Copyright 2019 Ryan Wick (rrwick@gmail.com)
# https://github.com/rrwick/August-2019-consensus-accuracy-update

# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should
# have received a copy of the GNU General Public License along with this program. If not, see
# <http://www.gnu.org/licenses/>.

# $1 = ref, $2 = pred

minimap2 -x asm5 -t 8 -c $1 $2 > minimapped.paf
python3 scripts/read_length_identity.py $2 minimapped.paf > identities.data
python3 scripts/medians.py identities.data
rm minimapped.paf
