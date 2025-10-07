# 2025-10-07T11:18:54.680978600
import vitis

client = vitis.create_client()
client.set_workspace(path="ff_bnn_stage")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

