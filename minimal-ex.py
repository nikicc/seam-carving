import pyopencl as cl
import numpy as np

class OpenCL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        self.program = cl.Program(self.ctx, fstr).build()

    def opencl_energy(self, img):
        mf = cl.mem_flags

        self.img = img.astype(np.float32)

        self.img_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.img)
        self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.img.nbytes)

        self.program.part1(self.queue, self.img.shape, None, self.img_buf, self.dest_buf)
        c = np.empty_like(self.img)
        cl.enqueue_read_buffer(self.queue, self.dest_buf, c).wait()
        return c

example = OpenCL()
example.loadProgram("get_energy.cl")
image = np.random.rand(320, 512, 4)
image = image.astype(np.float32)
results = example.opencl_energy(image)
print("All items are equal:", (results==image).all())