# 2025-10-23T18:29:34.376167500
import vitis

client = vitis.create_client()
client.set_workspace(path="ff_bnn_stage")

vitis.dispose()

