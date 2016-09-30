from dolfin import *
from mshr import *

base = Rectangle(dolfin.Point(0, 0),dolfin.Point(10, 10))
hole = Circle(dolfin.Point(5,5),2.5)
plot(generate_mesh(base - hole, 25))
interactive()


