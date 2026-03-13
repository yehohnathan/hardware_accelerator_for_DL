# 2026-03-12T16:10:47.307277200
import vitis

client = vitis.create_client()
client.set_workspace(path="ff_bnn_stage")

vitis.dispose()

