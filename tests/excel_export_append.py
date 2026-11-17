"""Append the real Customer Care export to an existing XLSX without rewriting it."""
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from zipfile import ZipFile

from lxml import etree as E
from openpyxl.utils import get_column_letter, range_boundaries

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
SHEET_ORDER=('sheetPr','dimension','sheetViews','sheetFormatPr','cols','sheetData','sheetCalcPr',
 'sheetProtection','protectedRanges','scenarios','autoFilter','sortState','dataConsolidate',
 'customSheetViews','mergeCells','phoneticPr','conditionalFormatting','dataValidations',
 'hyperlinks','printOptions','pageMargins','pageSetup','headerFooter','rowBreaks','colBreaks',
 'customProperties','cellWatches','ignoredErrors','smartTags','drawing','legacyDrawing',
 'legacyDrawingHF','picture','oleObjects','controls','webPublishItems','tableParts','extLst')
STYLE_ORDER=('numFmts','fonts','fills','borders','cellStyleXfs','cellXfs','cellStyles','dxfs','tableStyles','colors','extLst')


def _insert(root,node,order):
    name=E.QName(node).localname
    rank=order.index(name)
    position=next((i for i,c in enumerate(root)
        if E.QName(c).localname in order and order.index(E.QName(c).localname)>rank),len(root))
    root.insert(position,node)


def _key(node):
    return node.tag,tuple(sorted(node.attrib.items())),node.text,tuple(_key(c) for c in node)


def _collection(root,name):
    node=root.find(f'{{{NS}}}{name}')
    if node is None:
        node=E.Element(f'{{{NS}}}{name}',count='0');_insert(root,node,STYLE_ORDER)
    return node


def _styles(target,source):
    """Append/deduplicate export styles, preserving every existing style index."""
    maps={}
    source_formats=source.find(f'{{{NS}}}numFmts')
    target_formats=_collection(target,'numFmts')
    formats={n.get('formatCode'):int(n.get('numFmtId')) for n in target_formats}
    next_id=max([163,*formats.values()])+1
    maps['numFmtId']={}
    for item in source_formats if source_formats is not None else []:
        code=item.get('formatCode');old=int(item.get('numFmtId'))
        if code not in formats:
            clone=deepcopy(item);clone.set('numFmtId',str(next_id));target_formats.append(clone)
            formats[code]=next_id;next_id+=1
        maps['numFmtId'][old]=formats[code]
    target_formats.set('count',str(len(target_formats)))
    for name,reference in [('fonts','fontId'),('fills','fillId'),('borders','borderId'),
                           ('cellStyleXfs','xfId'),('cellXfs','style')]:
        dest=_collection(target,name)
        existing={_key(n):i for i,n in enumerate(dest)}
        source_items=source.find(f'{{{NS}}}{name}')
        mapping={}
        for i,item in enumerate(source_items if source_items is not None else []):
            clone=deepcopy(item)
            if name in {'cellStyleXfs','cellXfs'}:
                for key in ('fontId','fillId','borderId','numFmtId','xfId'):
                    if key in clone.attrib and key in maps:
                        old=int(clone.get(key));clone.set(key,str(maps[key].get(old,old)))
            key=_key(clone)
            if key not in existing:
                existing[key]=len(dest);dest.append(clone)
            mapping[i]=existing[key]
        dest.set('count',str(len(dest)));maps[reference]=mapping
    return maps['style']


def _member(z,sheet):
    root=E.fromstring(z.read('xl/workbook.xml'))
    rid=next(s.get(f'{{{REL}}}id') for s in root.find(f'{{{NS}}}sheets') if s.get('name')==sheet)
    relations=E.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    target=next(n.get('Target') for n in relations if n.get('Id')==rid)
    return target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl')/target)


def _column(address):
    value=0
    for c in re.match('[A-Z]+',address).group():value=value*26+ord(c)-64
    return value


def _xml(node):
    return E.tostring(node,encoding='UTF-8',xml_declaration=True,standalone=True)


def _split_columns(cols,start,end):
    for col in list(cols):
        lo,hi=int(col.get('min')),int(col.get('max'))
        if hi<start or lo>end:continue
        cols.remove(col)
        if lo<start:
            left=deepcopy(col);left.set('max',str(start-1));cols.append(left)
        if hi>end:
            right=deepcopy(col);right.set('min',str(end+1));cols.append(right)


def append_export(path,sheet,export_bytes,row_map,start_col=16,owned_width=0):
    """Copy the export's 61 columns, merged headers, widths, styles and links."""
    path=Path(path);width=61;end=start_col+width-1;offset=start_col-1
    with ZipFile(path) as target,ZipFile(BytesIO(export_bytes)) as exported:
        member=_member(target,sheet)
        root=E.fromstring(target.read(member))
        source=E.fromstring(exported.read('xl/worksheets/sheet1.xml'))
        data=root.find(f'{{{NS}}}sheetData')
        rows={int(r.get('r')):r for r in data}
        # Recover a completed workbook write if the JSON width checkpoint lagged it.
        # Both full header rows must match; a partial block or unrelated notes cannot
        # grant ownership of the remaining columns.
        def header_values(sheet_data):
            values={}
            for row in sheet_data:
                index=int(row.get('r'))
                if index>2:continue
                for cell in row:
                    strings=cell.xpath('s:is//s:t/text()',namespaces={'s':NS})
                    value=cell.find(f'{{{NS}}}v')
                    values[index,_column(cell.get('r'))]=''.join(strings) if strings else (value.text if value is not None else '')
            return values
        actual=header_values(data)
        expected=header_values(source.find(f'{{{NS}}}sheetData'))
        if all(actual.get((r,c+offset),'')==expected.get((r,c),'')
               for r in (1,2) for c in range(1,width+1)):
            owned_width=width
        for row in data:
            for cell in list(row):
                col=_column(cell.get('r'))
                if start_col<=col<=end:
                    has_value=cell.find(f'{{{NS}}}v') is not None or cell.find(f'{{{NS}}}f') is not None or bool(cell.xpath('.//s:t/text()',namespaces={'s':NS}))
                    if col>=start_col+owned_width and has_value:
                        raise ValueError(f'Append would overwrite an unrelated existing cell: {cell.get("r")}')
                    row.remove(cell)
        styles_root=E.fromstring(target.read('xl/styles.xml'))
        style_map=_styles(styles_root,E.fromstring(exported.read('xl/styles.xml')))
        map_row=lambda r:r if r<3 else row_map[r]
        map_ref=lambda ref:f'{get_column_letter(_column(ref)+offset)}{map_row(int(re.search(r"\d+",ref).group()))}'
        for source_row in source.find(f'{{{NS}}}sheetData'):
            index=map_row(int(source_row.get('r')))
            row=rows.get(index)
            if row is None:
                row=E.Element(f'{{{NS}}}row',r=str(index));rows[index]=row
                pos=next((i for i,r in enumerate(data) if int(r.get('r'))>index),len(data));data.insert(pos,row)
            if source_row.get('ht'):
                row.set('ht',str(max(float(row.get('ht','0')),float(source_row.get('ht')))));row.set('customHeight','1')
            for cell in source_row:
                clone=deepcopy(cell);clone.set('r',map_ref(cell.get('r')))
                clone.set('s',str(style_map[int(cell.get('s','0'))]))
                col=_column(clone.get('r'))
                pos=next((i for i,c in enumerate(row) if _column(c.get('r'))>col),len(row));row.insert(pos,clone)
        cols=root.find(f'{{{NS}}}cols')
        if cols is None:cols=E.Element(f'{{{NS}}}cols');_insert(root,cols,SHEET_ORDER)
        _split_columns(cols,start_col,end)
        for col in source.find(f'{{{NS}}}cols'):
            clone=deepcopy(col)
            clone.set('min',str(int(col.get('min'))+offset));clone.set('max',str(int(col.get('max'))+offset))
            if clone.get('style'):clone.set('style',str(style_map[int(clone.get('style'))]))
            cols.append(clone)
        cols[:]=sorted(cols,key=lambda c:int(c.get('min')))
        merges=root.find(f'{{{NS}}}mergeCells')
        if merges is None:merges=E.Element(f'{{{NS}}}mergeCells');_insert(root,merges,SHEET_ORDER)
        for merge in list(merges):
            lo,_,hi,_=range_boundaries(merge.get('ref'))
            if hi>=start_col and lo<=end:
                if lo<start_col or hi>end:raise ValueError('An existing merged range crosses the appended block')
                merges.remove(merge)
        for merge in source.find(f'{{{NS}}}mergeCells'):
            a,b=merge.get('ref').split(':');merges.append(E.Element(f'{{{NS}}}mergeCell',ref=f'{map_ref(a)}:{map_ref(b)}'))
        merges.set('count',str(len(merges)))
        # Preserve existing hyperlinks and add the export's conversation links.
        rel_path=str(PurePosixPath(member).parent/'_rels'/(PurePosixPath(member).name+'.rels'))
        relations=E.fromstring(target.read(rel_path)) if rel_path in target.namelist() else E.Element(f'{{{PKG}}}Relationships',nsmap={None:PKG})
        links=root.find(f'{{{NS}}}hyperlinks')
        if links is not None:
            removed=[]
            for link in list(links):
                if start_col<=_column(link.get('ref'))<=end:
                    removed.append(link.get(f'{{{REL}}}id'));links.remove(link)
            for rel in list(relations):
                if rel.get('Id') in removed:relations.remove(rel)
        source_links=source.find(f'{{{NS}}}hyperlinks')
        if source_links is not None:
            if links is None:links=E.Element(f'{{{NS}}}hyperlinks');_insert(root,links,SHEET_ORDER)
            source_relations=E.fromstring(exported.read('xl/worksheets/_rels/sheet1.xml.rels'))
            by_id={r.get('Id'):r for r in source_relations}
            used={r.get('Id') for r in relations}
            for link in source_links:
                new_id=f'rId{len(used)+1}'
                while new_id in used:new_id=f'rId{int(new_id[3:])+1}'
                used.add(new_id)
                rel=deepcopy(by_id[link.get(f'{{{REL}}}id')]);rel.set('Id',new_id);relations.append(rel)
                clone=deepcopy(link);clone.set('ref',map_ref(link.get('ref')));clone.set(f'{{{REL}}}id',new_id);links.append(clone)
        if links is not None and not len(links):root.remove(links)
        max_col=max(_column(c.get('r')) for r in data for c in r)
        dimension=root.find(f'{{{NS}}}dimension')
        if dimension is not None:dimension.set('ref',f'A1:{get_column_letter(max_col)}{max(rows)}')
        replacements={member:_xml(root),'xl/styles.xml':_xml(styles_root)}
        if len(relations) or rel_path in target.namelist():replacements[rel_path]=_xml(relations)
        fd,temp=tempfile.mkstemp(suffix='.xlsx',dir=path.parent);os.close(fd)
        try:
            with ZipFile(temp,'w') as output:
                for info in target.infolist():output.writestr(info,replacements.pop(info.filename,target.read(info.filename)))
                for name,payload in replacements.items():output.writestr(name,payload)
            shutil.copymode(path,temp)
        except BaseException:
            Path(temp).unlink(missing_ok=True);raise
    try:os.replace(temp,path)
    finally:Path(temp).unlink(missing_ok=True)
