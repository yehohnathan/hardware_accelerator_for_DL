# 2026-03-16T14:11:13.521037200
import vitis

client = vitis.create_client()
client.set_workspace(path="ff_bnn_stage")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

vitis.dispose()

