# 2026-03-09T06:32:15.056677800
import vitis

client = vitis.create_client()
client.set_workspace(path="ff_bnn_stage")

vitis.dispose()

