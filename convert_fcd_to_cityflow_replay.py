import xml.etree.ElementTree as ET
import math
import sys
import argparse


def convert(fcd_xml_path, output_txt_path,
            x_offset=300.0, y_offset=300.0,
            default_length=5.0, default_width=2.0):
    """
    Convert SUMO FCD XML output to CityFlow replay .txt format.

    SUMO shifts coordinates to be all-positive (netOffset in roadnet.net.xml).
    CityFlow expects the original centered coordinates.
    So we subtract the netOffset (default 300, 300) from every vehicle's x, y.

    Format per timestep:
        x y angle vehicle_id 0 length width,<more vehicles>;<tls_states>

    TLS states are left empty here — signals will render in default state.
    """
    timesteps_written = 0
    with open(output_txt_path, 'w') as out:
        try:
            for event, elem in ET.iterparse(fcd_xml_path, events=('end',)):
                if elem.tag != 'timestep':
                    continue

                vehicle_records = []
                for v in elem.findall('vehicle'):
                    # SUMO coords -> CityFlow coords (subtract netOffset)
                    x = float(v.attrib['x']) - x_offset
                    y = float(v.attrib['y']) - y_offset

                    # SUMO angle (degrees, 0=north, clockwise) -> CityFlow (radians, math convention)
                    angle_deg = float(v.attrib['angle'])
                    angle_rad = math.radians(90.0 - angle_deg)
                    while angle_rad < 0:
                        angle_rad += 2 * math.pi
                    while angle_rad >= 2 * math.pi:
                        angle_rad -= 2 * math.pi

                    vid = v.attrib['id']
                    status = "0"
                    vehicle_records.append(
                        f"{x} {y} {angle_rad} {vid} {status} {default_length} {default_width}"
                    )

                # TLS section left empty for now (signals show in default state)
                tls_str = ""
                line = ",".join(vehicle_records) + ";" + tls_str
                out.write(line + "\n")
                timesteps_written += 1

                # Free memory as we go
                elem.clear()
        except ET.ParseError as e:
            print(f"Warning: parse error at end of file (likely truncated): {e}")
            print(f"Successfully wrote {timesteps_written} timesteps before the error.")

    print(f"Converted {fcd_xml_path} -> {output_txt_path} ({timesteps_written} timesteps)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert SUMO FCD XML to CityFlow replay format."
    )
    parser.add_argument("fcd_xml", help="Input FCD XML file (from SUMO --fcd-output)")
    parser.add_argument("output_txt", help="Output CityFlow replay .txt file")
    parser.add_argument("--x-offset", type=float, default=300.0,
                        help="X offset to subtract (from netOffset in roadnet.net.xml; default 300.0)")
    parser.add_argument("--y-offset", type=float, default=300.0,
                        help="Y offset to subtract (from netOffset in roadnet.net.xml; default 300.0)")
    parser.add_argument("--length", type=float, default=5.0, help="Default vehicle length (default 5.0)")
    parser.add_argument("--width", type=float, default=2.0, help="Default vehicle width (default 2.0)")

    args = parser.parse_args()
    convert(
        fcd_xml_path=args.fcd_xml,
        output_txt_path=args.output_txt,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        default_length=args.length,
        default_width=args.width,
    )

