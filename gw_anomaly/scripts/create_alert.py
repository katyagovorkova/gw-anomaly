'''

Write down json in the following format:
{
    "gpstime": 1368221446,
    "detection_statistic": 8.87,
    "far": 1.1574074074074073e-05,
    "ifos": [
        "H1",
        "L1"
    ]
}

'''
import json

# read input data form the playground

# run GWAK on the input

# compute final metric on the input
alert = dict()
alert['gpstime'] = 1368221446
alert['detection_statistic'] = 8.87
alert['far'] = 1.1574074074074073e-05
alert['ifos'] = [
                'H1',
                'L1'
                ]

with open(f'alert_{alert["gpstime"]}.json', 'w') as f:
  json.dump(alert, f)

### How a-frame does it
# class Trigger:
#     def __init__(self, server: Gdb, write_dir: Path) -> None:
#         self.write_dir = write_dir

#         if server in ["playground", "test"]:
#             server = f"https://gracedb-{server}.ligo.org/api/"
#         elif server == "production":
#             server = "https://gracedb.ligo.org/api/"
#         elif server == "local":
#             self.gdb = LocalGdb()
#             return
#         else:
#             raise ValueError(f"Unknown server {server}")
#         self.gdb = GraceDb(service_url=server)

#     def __post_init__(self):
#         self.write_dir.mkdir(exist_ok=True, parents=True)

#     def submit(
#         self,
#         event: Event,
#         ifos: List[str],
#         datadir: Path,
#         ifo_suffix: str = None,
#     ):
#         gpstime = event.gpstime
#         event_dir = self.write_dir / f"event_{int(gpstime)}"
#         event_dir.mkdir(exist_ok=True, parents=True)
#         filename = event_dir / f"event-{int(gpstime)}.json"

#         event = asdict(event)
#         event["ifos"] = ifos
#         filecontents = str(event)
#         filecontents = json.loads(filecontents.replace("'", '"'))
#         with open(filename, "w") as f:
#             json.dump(filecontents, f)

#         logging.info(f"Submitting trigger to file {filename}")
#         response = self.gdb.createEvent(
#             group="CBC",
#             pipeline="aframe",
#             filename=str(filename),
#             search="AllSky",
#         )
#         submission_time = float(tconvert(datetime.now(tz=timezone.utc)))
#         t_write = get_frame_write_time(gpstime, datadir, ifos, ifo_suffix)
#         # Time to submit since event occured and since the file was written
#         total_latency = submission_time - gpstime
#         write_latency = t_write - gpstime
#         aframe_latency = submission_time - t_write

#         latency_fname = event_dir / "latency.log"
#         latency = "Total Latency (s),Write Latency (s),Aframe Latency (s)\n"
#         latency += f"{total_latency},{write_latency},{aframe_latency}"
#         with open(latency_fname, "w") as f:
#             f.write(latency)

#         return response