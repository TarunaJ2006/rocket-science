"""rocketScience.py

Small utility to evaluate trapezoidal fin geometries for a model rocket and
search for configurations that maximise a simple nondimensional stability margin.

This script implements helper functions for trapezoidal fin geometry (area,
mean aerodynamic chord), a simple centre-of-pressure estimate and a configurable
grid search over geometric parameters.

Usage (simple):
	"""rocketScience.py

	Small utility to evaluate trapezoidal fin geometries for a model rocket and
	search for configurations that maximise a simple nondimensional stability margin.

	This script implements helper functions for trapezoidal fin geometry (area,
	mean aerodynamic chord), a simple centre-of-pressure estimate and a configurable
	grid search over geometric parameters.

	Usage (simple):
		python rocketScience.py

	Usage (advanced):
		python rocketScience.py --root-range 0.04 0.07 --tip-range 0.01 0.05 --steps 8

	The algorithm and parameter defaults are chosen to match the repository README
	and to be easy to read and adapt for experiments. This is an engineering demo,
	not a production-grade aerodynamics tool.
	"""

	from __future__ import annotations

	import argparse
	from dataclasses import dataclass
	import math
	import logging
	from typing import Tuple

	import numpy as np


	@dataclass
	class FinGeometry:
		"""Container for a trapezoidal fin geometry specification.

		Attributes
		----------
		root_chord : float
			Root chord length in meters.
		tip_chord : float
			Tip chord length in meters.
		span : float
			Fin span (height) in meters.
		sweep_deg : float
			Sweep angle in degrees.
		"""
		root_chord: float  # m
		tip_chord: float   # m
		span: float        # m
		sweep_deg: float   # degrees


	def trapezoid_area(root_chord: float, tip_chord: float, span: float) -> float:
		"""Compute the planform area of a trapezoidal fin.

		area = (root + tip) / 2 * span
		"""
		return (root_chord + tip_chord) * 0.5 * span


	def mean_aerodynamic_chord(root_chord: float, tip_chord: float, span: float) -> float:
		"""Compute the Mean Aerodynamic Chord (MAC) for a trapezoid.

		Formula for MAC (x-bar dimension across chord):
			MAC = (2/3) * (root^2 + root*tip + tip^2) / (root + tip)

		Note: the returned value is a chord length (m).
		"""
		r = root_chord
		t = tip_chord
		if (r + t) == 0:
			return 0.0
		mac = (2.0 / 3.0) * (r ** 2 + r * t + t ** 2) / (r + t)
		return mac


	def cp_from_root_le(root_chord: float, tip_chord: float, span: float, sweep_deg: float) -> float:
		"""Estimate the fin centre-of-pressure (CP) measured from the fin root leading edge.

		This uses a simplified approach:
		  - baseline CP: 0.75 * MAC measured from the root leading edge
		  - a simple sweep-induced aft shift approximated as 0.5 * span * tan(sweep)

		The sweep shift is heuristic and included to illustrate the parameter effect.
		Returns CP position in meters from the fin root leading edge.
		"""
		mac = mean_aerodynamic_chord(root_chord, tip_chord, span)
		baseline_cp = 0.75 * mac
		sweep_rad = math.radians(sweep_deg)
		# Simple heuristic for sweep shift (user should treat this as illustrative):
		sweep_shift = 0.5 * span * math.tan(sweep_rad)
		cp = baseline_cp + 0.5 * sweep_shift
		return cp


	def nondimensional_stability_margin(cp_from_nose: float, cg_from_nose: float, rocket_diameter: float) -> float:
		"""Compute a nondimensional stability margin: (CP - CG) / diameter.

		Higher positive values generally indicate greater static stability for small
		perturbations in classical rocket stability reasoning (but this simple metric
		ignores many real-world effects).
		"""
		return (cp_from_nose - cg_from_nose) / rocket_diameter


	def grid_search_best(
		cg_from_nose: float,
		rocket_diameter: float,
		fin_root_pos: float,
		root_range: Tuple[float, float],
		tip_range: Tuple[float, float],
		span_range: Tuple[float, float],
		sweep_range: Tuple[float, float],
		steps: int = 6,
	) -> Tuple[FinGeometry, float]:
		"""Brute-force grid search over the provided parameter ranges.

		Returns the best FinGeometry and its nondimensional stability margin.
		"""
		best_margin = -float('inf')
		best_geom = None

		roots = np.linspace(root_range[0], root_range[1], steps)
		tips = np.linspace(tip_range[0], tip_range[1], steps)
		spans = np.linspace(span_range[0], span_range[1], steps)
		sweeps = np.linspace(sweep_range[0], sweep_range[1], steps)

		for r in roots:
			for t in tips:
				for s in spans:
					for sw in sweeps:
						cp_local = cp_from_root_le(r, t, s, sw)
						# Convert CP measured from fin root LE to CP from nose:
						cp_from_nose = fin_root_pos + cp_local
						margin = nondimensional_stability_margin(cp_from_nose, cg_from_nose, rocket_diameter)
						if margin > best_margin:
							best_margin = margin
							best_geom = FinGeometry(root_chord=r, tip_chord=t, span=s, sweep_deg=sw)

		return best_geom, best_margin


	def parse_args() -> argparse.Namespace:
		p = argparse.ArgumentParser(description="Rocket fin stability grid search utility")
		p.add_argument('--cg', type=float, default=0.15, help='CG location from nose (m)')
		p.add_argument('--diameter', type=float, default=0.05, help='Rocket body diameter (m)')
		p.add_argument('--fin-root-pos', type=float, default=0.2, help='Fin root leading edge axial position from nose (m)')
		p.add_argument('--steps', type=int, default=6, help='Grid steps per parameter')
		p.add_argument('--root-range', nargs=2, type=float, default=[0.05, 0.06], help='Root chord range (m)')
		p.add_argument('--tip-range', nargs=2, type=float, default=[0.02, 0.04], help='Tip chord range (m)')
		p.add_argument('--span-range', nargs=2, type=float, default=[0.03, 0.04], help='Span range (m)')
		p.add_argument('--sweep-range', nargs=2, type=float, default=[15.0, 20.0], help='Sweep angle range (deg)')
		return p.parse_args()


	def main():
		args = parse_args()

		best_geom, best_margin = grid_search_best(
			cg_from_nose=args.cg,
			rocket_diameter=args.diameter,
			fin_root_pos=args.fin_root_pos,
			root_range=(args.root_range[0], args.root_range[1]),
			tip_range=(args.tip_range[0], args.tip_range[1]),
			span_range=(args.span_range[0], args.span_range[1]),
			sweep_range=(args.sweep_range[0], args.sweep_range[1]),
			steps=args.steps,
		)

		logging.info("Best fin geometry found:")
		logging.info(f"  Root chord: {best_geom.root_chord:.4f} m")
		logging.info(f"  Tip chord : {best_geom.tip_chord:.4f} m")
		logging.info(f"  Span      : {best_geom.span:.4f} m")
		logging.info(f"  Sweep     : {best_geom.sweep_deg:.2f} deg")
		logging.info(f"Nondimensional stability margin: {best_margin:.4f}"
					 "  ( (CP - CG)/diameter )")


	if __name__ == '__main__':
		# Configure basic logging for CLI usage
		logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
		main()
		tip_range: Tuple[float, float],
		span_range: Tuple[float, float],
		sweep_range: Tuple[float, float],
		steps: int = 6,
	) -> Tuple[FinGeometry, float]:
		"""Brute-force grid search over the provided parameter ranges.

		Returns the best FinGeometry and its nondimensional stability margin.
		"""
		best_margin = -float('inf')
		best_geom = None

		roots = np.linspace(root_range[0], root_range[1], steps)
		tips = np.linspace(tip_range[0], tip_range[1], steps)
		spans = np.linspace(span_range[0], span_range[1], steps)
		sweeps = np.linspace(sweep_range[0], sweep_range[1], steps)

		for r in roots:
			for t in tips:
				for s in spans:
					for sw in sweeps:
						cp_local = cp_from_root_le(r, t, s, sw)
						# Convert CP measured from fin root LE to CP from nose:
						cp_from_nose = fin_root_pos + cp_local
						margin = nondimensional_stability_margin(cp_from_nose, cg_from_nose, rocket_diameter)
						if margin > best_margin:
							best_margin = margin
							best_geom = FinGeometry(root_chord=r, tip_chord=t, span=s, sweep_deg=sw)

		return best_geom, best_margin


	def parse_args() -> argparse.Namespace:
		p = argparse.ArgumentParser(description="Rocket fin stability grid search utility")
		p.add_argument('--cg', type=float, default=0.15, help='CG location from nose (m)')
		p.add_argument('--diameter', type=float, default=0.05, help='Rocket body diameter (m)')
		p.add_argument('--fin-root-pos', type=float, default=0.2, help='Fin root leading edge axial position from nose (m)')
		p.add_argument('--steps', type=int, default=6, help='Grid steps per parameter')
		p.add_argument('--root-range', nargs=2, type=float, default=[0.05, 0.06], help='Root chord range (m)')
		p.add_argument('--tip-range', nargs=2, type=float, default=[0.02, 0.04], help='Tip chord range (m)')
		p.add_argument('--span-range', nargs=2, type=float, default=[0.03, 0.04], help='Span range (m)')
		p.add_argument('--sweep-range', nargs=2, type=float, default=[15.0, 20.0], help='Sweep angle range (deg)')
		return p.parse_args()


	def main():
		args = parse_args()

		best_geom, best_margin = grid_search_best(
			cg_from_nose=args.cg,
			rocket_diameter=args.diameter,
			fin_root_pos=args.fin_root_pos,
			root_range=(args.root_range[0], args.root_range[1]),
			tip_range=(args.tip_range[0], args.tip_range[1]),
			span_range=(args.span_range[0], args.span_range[1]),
			sweep_range=(args.sweep_range[0], args.sweep_range[1]),
			steps=args.steps,
		)

		logging.info("Best fin geometry found:")
		logging.info(f"  Root chord: {best_geom.root_chord:.4f} m")
		logging.info(f"  Tip chord : {best_geom.tip_chord:.4f} m")
		logging.info(f"  Span      : {best_geom.span:.4f} m")
		logging.info(f"  Sweep     : {best_geom.sweep_deg:.2f} deg")
		logging.info(f"Nondimensional stability margin: {best_margin:.4f}"
					 "  ( (CP - CG)/diameter )")


	if __name__ == '__main__':
		# Configure basic logging for CLI usage
		logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
		main()

