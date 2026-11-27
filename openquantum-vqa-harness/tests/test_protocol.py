"""Exercise the real stdio transport used by both supported agent hosts."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ProtocolTests(unittest.TestCase):
    def test_compile_through_actual_mcp(self):
        asyncio.run(self.exercise())

    async def exercise(self):
        packet = json.loads((Path(__file__).parents[1] / "examples/three-spin-vqe.json").read_text())
        packet["vqa"] = {"parameter_grid": [{"theta": 0.0}, {"theta": 0.6}],
                         "shots_per_term": 16, "terminal_shots_per_term": 32, "seed": 41}
        server = StdioServerParameters(command=sys.executable, args=["-m", "vqa_harness.server"])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "job"
            async with stdio_client(server) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.assertEqual({t.name for t in tools.tools},
                                     {"harness_status", "compile_vqa_job", "compile_native_host_policy", "compile_joint_job"})
                    region = {"interface": "cz-rotation-coherent",
                              "capabilities": {"individual_u3": True, "addressed_row_hrzhrx": False},
                              "settings": [{"binding": 0, "basis": "Z", "source": {
                                  "num_qubits": 1, "gates": [
                                      {"gate": "u1", "qubits": [0], "params": [.1]},
                                      {"gate": "u1", "qubits": [0], "params": [-.1]}]}}]}
                    native = await session.call_tool("compile_native_host_policy", {"job": region})
                    self.assertFalse(native.isError)
                    compiled = json.loads(native.content[0].text)
                    self.assertEqual(compiled["settings"][0]["policy"]["native_program"]["gates"], [])
                    reply = await session.call_tool("compile_vqa_job", {
                        "job": packet, "output_directory": str(out), "backend": "standalone"})
                    self.assertFalse(reply.isError)
                    saved = json.loads((out / "result.json").read_text())
                    summary = reply.structuredContent or json.loads(
                        next(item.text for item in reply.content if item.type == "text"))
                    self.assertEqual(summary["answer"], saved["answer"])
                    self.assertTrue(summary["verification"]["all_ideal_pauli_checks_passed"])
                    repeated = await session.call_tool("compile_vqa_job", {
                        "job": packet, "output_directory": str(out), "backend": "standalone"})
                    self.assertTrue(repeated.isError)
                    self.assertEqual(saved, json.loads((out / "result.json").read_text()))


if __name__ == "__main__":
    unittest.main()
