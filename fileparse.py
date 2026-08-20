import logging
import csv

log = logging.getLogger(__name__)

def parse(lines, select=None, types=None, has_headers=True, delimiter=",",
          silence_errors = False) :
    
    # create records list by parsing and type transformation of csv file.

    if select and not has_headers:
        raise RuntimeError('select requires column headers')
    rows = csv.reader(lines,delimiter=delimiter)

    headers = next(rows) if has_headers else []

    if select :
        indices = [headers.index(colname) for colname in select]
        headers = select

    records = []

    for rowno, row in enumerate(rows,1) :
        if not row:
            continue
        if select :
            row = [row[index] for index in indices]

        if types :
            try :
                row = [func(val) for func, val in zip(types,row)]
            except ValueError as e:
                if not silence_errors:
                    log.warning("Row %d: couldnt convert %s", rowno, row)
                    log.debug("Row %d: Reason %s", rowno, e)
                    continue
        
        if headers: 
            record = dict(zip(headers,row))
        else :
            record = tuple(row)
        records.append(record)

    return records
        