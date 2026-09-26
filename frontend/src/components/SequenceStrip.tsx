import { memo, useEffect, useRef, useState } from 'react';
import { t } from '../i18n';
import { AA_INFO, GROUP_COLOR, llrColor, plddtColor } from '../workbench';

export type StripColor = 'group' | 'plddt' | 'tolerance' | 'none';

interface Props {
    sequence: string;
    mutated?: Set<number>;
    selected?: number | null;
    plddt?: number[];
    tolerance?: number[];
    color: StripColor;
    onClick?: (pos: number) => void;
    onHover?: (pos: number | null) => void;
    perRow?: number;
}

function residueColor(ch: string, i: number, props: Props): string | undefined {
    if (props.color === 'group') return GROUP_COLOR[AA_INFO[ch]?.group ?? ''] ?? '#6b7280';
    if (props.color === 'plddt' && props.plddt?.[i] !== undefined) return plddtColor(props.plddt[i]);
    if (props.color === 'tolerance' && props.tolerance?.[i] !== undefined) return llrColor(props.tolerance[i] * 1.5);
    return undefined;
}

/** Width of 10 residues: 10 cells of 12px + 0.5px gaps, plus the wider gap after every 10th. */
const BLOCK_PX = 128.5;
const NUMBER_PX = 36;

export const SequenceStrip = memo(function SequenceStrip(props: Props) {
    const { sequence, mutated, selected, onClick, onHover } = props;
    const ref = useRef<HTMLDivElement>(null);
    const [fit, setFit] = useState(30);
    // wrap in blocks of 10 that fit the panel instead of overflowing it
    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const measure = () => setFit(Math.max(10, Math.floor((el.clientWidth - NUMBER_PX) / BLOCK_PX) * 10));
        measure();
        const ro = new ResizeObserver(measure);
        ro.observe(el);
        return () => ro.disconnect();
    }, []);
    const perRow = props.perRow ?? fit;
    const rows: number[] = [];
    for (let i = 0; i < sequence.length; i += perRow) rows.push(i);
    return (
        <div className="seqstrip" ref={ref} onMouseLeave={() => onHover?.(null)}>
            {rows.map(start => (
                <div className="seq-row" key={start}>
                    <span className="seq-num">{start + 1}</span>
                    <span className="seq-cells">
                        {[...sequence.slice(start, start + perRow)].map((ch, k) => {
                            const i = start + k;
                            const pos = i + 1;
                            const bg = residueColor(ch, i, props);
                            const cls = [
                                'seq-cell',
                                mutated?.has(pos) ? 'mut' : '',
                                selected === pos ? 'sel' : '',
                                pos % 10 === 0 ? 'tick' : '',
                            ].join(' ');
                            const title = `${ch}${pos} ${AA_INFO[ch]?.name ?? ''}${props.plddt?.[i] !== undefined ? ` pLDDT ${props.plddt[i].toFixed(0)}` : ''}${props.tolerance?.[i] !== undefined ? ` ${t('許容度')} ${props.tolerance[i].toFixed(2)}` : ''}`;
                            return (
                                <span key={i} className={cls} title={title}
                                    style={bg ? { background: bg, color: '#0b0f14' } : undefined}
                                    onMouseEnter={() => onHover?.(pos)}
                                    onClick={() => onClick?.(pos)}>
                                    {ch}
                                </span>
                            );
                        })}
                    </span>
                </div>
            ))}
        </div>
    );
});
