"""Table generation for the model transformation process."""

import copy
import csv
from warnings import warn
from typing import List, Iterable, Tuple
from dataclasses import dataclass, asdict, field

from sysopt.problems.problem_data import Domain
from sysopt.modelling.block import (
    Block, Port, Connection, Composite, Channel
)
from sysopt import warnings
from sysopt import exceptions
from sysopt.helpers import partition


@dataclass
class TableEntry:
    """Name and index of a block variable."""
    local_name: str
    block: str
    local_index: int
    global_index: int

    @property
    def name(self):
        return f'{self.block}/{self.local_name}'

    def __str__(self):
        return f'\"{self.name}\", {self.global_index} -> {self.local_index})\n'

    def __repr__(self):
        return f'{self.global_index} -> ({self.name}, {self.local_index})\n'


@dataclass
class WireEntry:
    """The local (port, channel) and global indices for and internal wire."""
    source_port: str
    source_channel: int
    destination_port: str
    destination_channel: str
    source_index: int
    destination_index: int


@dataclass
class Tables:
    """Database of all variables required for a flattened model."""
    states: List[TableEntry] = field(default_factory=list)
    constraints: List[TableEntry] = field(default_factory=list)
    inputs: List[TableEntry] = field(default_factory=list)
    outputs: List[TableEntry] = field(default_factory=list)
    parameters: List[TableEntry] = field(default_factory=list)
    wires: List[WireEntry] = field(default_factory=list)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, item):
        return item in self.keys()

    @staticmethod
    def keys():
        return {'states', 'constraints', 'inputs', 'outputs',
                'parameters', 'wires'}

    def __repr__(self):
        fields, entries = self.to_fields_and_entries()

        widths = (max(len(e) for e in l) for l in zip(*entries))
        widths = tuple(max(w, len(header)) for w, header in zip(widths, fields))

        text = '|'.join(
            f' {name: <{width}} '
            for name, width in zip(fields, widths)
        )
        text += '\n' + '+'.join('-' * (width + 2) for width in widths) + '\n'
        text += '\n'.join(
            '|'.join(f' {name: <{width}} '
                     for name, width in zip(entry, widths)
                     )
            for entry in entries
        )
        return text

    def to_fields_and_entries(self):
        fields = ['global variable', 'block', 'local variable',
                  'name', 'connected to']
        values = []
        netlist = {}

        def is_root(block_name: str):
            return len(block_name.split('/')) == 1

        for table in self.keys() - {'wires', 'inputs', 'outputs'}:
            for entry in self[table]:
                var_name = f'{table}_{entry.global_index}'
                values.append((
                    var_name,
                    f'{entry.block}',
                    f'{table}_{entry.local_index}',
                    entry.local_name,
                    ''
                ))

        external_ins, wired_inputs = partition(
            self.inputs, lambda e: is_root(e.block)
        )
        external_outs, wired_outputs = partition(
            self.outputs, lambda e: is_root(e.block)
        )

        for wire in self.wires:
            out_name = f'outputs_{wire.source_index}'
            in_name = f'inputs_{wire.destination_index}'
            netlist[out_name] = in_name
            netlist[in_name] = out_name

        for external_out in external_outs:
            out_name = f'outputs_{external_out.global_index}'
            target = f'{external_out.block}/outputs_{external_out.local_index}'
            netlist[out_name] = target

        for external_in in external_ins:
            in_name = f'outputs_{external_in.global_index}'
            target = f'{external_in.block}/outputs_{external_in.local_index}'
            netlist[in_name] = target

        for table, entries in (('inputs', wired_inputs),
                               ('outputs', wired_outputs)):
            for entry in entries:

                var_name = f'{table}_{entry.global_index}'

                if len(entry.block.split('/')) == 1:
                    connection = 'external'
                else:
                    try:
                        connection = netlist[var_name]
                    except KeyError:
                        connection = ''
                values.append(
                    (var_name,
                     f'{entry.block}',
                     f'{table}_{entry.local_index}',
                     entry.local_name,
                     connection
                ))

        return fields, values

    def to_csv(self, filename):
        fields, rows = self.to_fields_and_entries()
        with open(filename, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fields)
            writer.writerows(rows)


def create_tables_from_block(block: Block) -> Tables:

    tables = Tables()

    for attribute, attr_object in asdict(block.metadata).items():
        if attr_object:
            tables[attribute] = [
                TableEntry(block=str(block), local_name=name,
                           local_index=i, global_index=i)
                for i, name in enumerate(attr_object)
            ]
        else:
            tables[attribute] = []

    return tables


def merge_table(table_1: Tables, table_2: Tables) -> Tables:
    out_table = Tables()

    for key in set(table_1.keys()) | set(table_2.keys()):

        l1 = table_1[key] if key in table_1 else []
        l2 = table_2[key] if key in table_2 else []

        out_table[key] = [copy.copy(entry) for entry in l1]
        offset = global_dimension_of(l1)

        out_table[key] += [
            TableEntry(local_name=entry.local_name,
                       block=entry.block,
                       local_index=entry.local_index,
                       global_index=entry.global_index + offset)
            for entry in l2
        ]
    return out_table


def create_tables_from_blocks(*blocks):

    tables = [create_tables_from_block(block) for block in blocks]

    base, *rest = tables

    for table in rest:
        base = merge_table(base, table)

    return base


def global_dimension_of(table):
    if not table:
        return 0
    return 1 + max(entry.global_index for entry in table)


def find_channel_in_table(table: List[TableEntry],
                          port: Port,
                          local_index: int) -> int:

    def key(table_entry: TableEntry):
        return (table_entry.block == str(port.block)
                and table_entry.local_index == local_index)

    try:
        entry, = list(filter(key, table))
    except ValueError as ex:
        if len(list(filter(key, table))) == 0:
            message = f'Could not find {port.block}' \
                      f'[{local_index}] in table\n'
        else:
            message = f'Multiple entries for {port.block}' \
                      f'[{local_index}] in table: {table}\n'
        raise ValueError(message) from ex

    return entry.global_index


def is_internal(wire: Connection) -> bool:
    src, dest = wire
    return src.block.parent == dest.block.parent


def get_ports_and_indices(wire: Connection) -> \
       Iterable[Tuple[Channel, Channel]]:

    src, dest = wire

    index_pairing = list(zip(src.indices, dest.indices))
    src_port = src.port if isinstance(src, Channel) else src
    dest_port = dest.port if isinstance(dest, Channel) else dest

    return src_port, dest_port, index_pairing


def internal_wire_to_table_entries(tables: Tables,
                                   wire: Connection) -> List[WireEntry]:
    # for the source
    src_port, dest_port, indices = get_ports_and_indices(wire)
    entries = []
    for src_i, dest_i in indices:
        try:
            src_index = find_channel_in_table(
                tables.outputs, src_port, src_i
            )
        except ValueError as ex:
            raise exceptions.InternalWireNotFound(wire, *ex.args) from ex
        try:
            dest_index = find_channel_in_table(
                tables.inputs, dest_port, dest_i
            )
        except ValueError:
            warn(warnings.UnconnectedInput(wire))
            continue

        entries.append(WireEntry(
            source_port=str(src_port),
            destination_port=str(dest_port),
            source_channel=src_i,
            destination_channel=dest_i,
            source_index=src_index,
            destination_index=dest_index
        ))
    return entries


def is_wire_forwarding_inputs(wire: Connection) -> bool:
    src, dest = wire
    return src.port_type == dest.port_type == 'inputs'


def forwarded_input_to_table_entry(tables: Tables,
                                   wire: Connection) -> List[TableEntry]:

    src_port, dest_port, indices = get_ports_and_indices(wire)

    entries = []
    for src_i, dest_i in indices:
        global_index = find_channel_in_table(
            tables.inputs, dest_port, dest_i)

        entries.append(TableEntry(
            local_name=src_port.channel_name(src_i),
            block=str(src_port.block),
            local_index=src_i,
            global_index=global_index
        ))
    return entries


def forwarded_output_to_table_entry(tables: Tables,
                                    wire: Connection) -> List[TableEntry]:
    src_port, dest_port, indices = get_ports_and_indices(wire)
    entries = []
    for src_i, dest_i in indices:
        global_index = find_channel_in_table(tables.outputs, src_port, src_i)
        entries.append(TableEntry(
            local_name=dest_port.channel_name(dest_i),
            block=str(dest_port.block),
            local_index=dest_i,
            global_index=global_index
        ))
    return entries


def create_tables(all_blocks: List[Block]) -> Tables:

    tables = create_tables_from_blocks(
        *filter(lambda b: not isinstance(b, Composite), all_blocks)
    )

    trunks = list(filter(lambda b: isinstance(b, Composite), all_blocks))
    sizes = [
        global_dimension_of(tables[name])
        for name in ('states', 'constraints', 'inputs', 'parameters')
    ]

    domain = Domain(1, *sizes)

    tables.wires = []
    while trunks:
        next_block: Composite = trunks.pop()
        for wire in next_block.wires:
            if is_internal(wire):
                tables.wires += internal_wire_to_table_entries(tables, wire)
            elif is_wire_forwarding_inputs(wire):
                entries = forwarded_input_to_table_entry(tables, wire)
                tables.inputs += entries
            else:
                entries = forwarded_output_to_table_entry(tables, wire)
                tables.outputs += entries

    return tables, domain


