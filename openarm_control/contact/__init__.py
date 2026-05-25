"""Contact-rich / compliant control for the OpenArm (extension phase F2).

OpenArm's defining trait is backdrivable, compliant, contact-rich operation. This
package adds compliant control on top of the existing position-control stack
without touching it: :class:`AdmittanceController` reads the external force at the
gripper and softens the Cartesian reference so the arm yields on contact instead
of pushing rigidly through. It is the basis for force-guarded insertion, surface
wiping, and operating the articulated fixtures (drawer/door/valve) in later phases.
"""

from .admittance import AdmittanceController, ee_contact_force

__all__ = ["AdmittanceController", "ee_contact_force"]
