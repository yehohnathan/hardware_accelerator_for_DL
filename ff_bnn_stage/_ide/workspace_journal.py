# 2025-09-15T09:54:08.597725500
import vitis

client = vitis.create_client()
client.set_workspace(path="ff_bnn_stage")

comp = client.create_hls_component(name = "hls_component",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

