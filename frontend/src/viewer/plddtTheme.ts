import { Bond, StructureElement, Unit } from 'molstar/lib/mol-model/structure';
import { t } from '../i18n';
import type { Location } from 'molstar/lib/mol-model/location';
import { ColorTheme } from 'molstar/lib/mol-theme/color';
import { ColorThemeCategory } from 'molstar/lib/mol-theme/color/categories';
import type { ThemeDataContext } from 'molstar/lib/mol-theme/theme';
import { Color } from 'molstar/lib/mol-util/color';
import { ParamDefinition as PD } from 'molstar/lib/mol-util/param-definition';

/**
 * AlphaFold-style pLDDT colouring read from the B-factor column, which is where Boltz-2
 * and AlphaFold DB store per-atom pLDDT (0–100).
 */
const VERY_HIGH = Color(0x0053d6);
const HIGH = Color(0x65cbf3);
const LOW = Color(0xffdb13);
const VERY_LOW = Color(0xff7d45);
const FALLBACK = Color(0x999999);

const Params = {};
type Params = typeof Params;

function bfactor(unit: Unit, element: number): number | undefined {
    if (!Unit.isAtomic(unit)) return undefined;
    const col = unit.model.atomicConformation.B_iso_or_equiv;
    return col.isDefined ? col.value(element) : undefined;
}

function colorFor(v: number | undefined): Color {
    if (v === undefined) return FALLBACK;
    if (v >= 90) return VERY_HIGH;
    if (v >= 70) return HIGH;
    if (v >= 50) return LOW;
    return VERY_LOW;
}

function PlddtBFactorColorTheme(_ctx: ThemeDataContext, props: PD.Values<Params>): ColorTheme<Params> {
    const color = (location: Location): Color => {
        if (StructureElement.Location.is(location)) return colorFor(bfactor(location.unit, location.element));
        if (Bond.isLocation(location)) {
            return colorFor(bfactor(location.aUnit, location.aUnit.elements[location.aIndex]));
        }
        return FALLBACK;
    };
    return {
        factory: PlddtBFactorColorTheme,
        granularity: 'group',
        color,
        props,
        description: t('pLDDT (B-factor 列)'),
        legend: {
            kind: 'table-legend',
            table: [
                [t('> 90 とても高い'), VERY_HIGH],
                [t('70–90 高い'), HIGH],
                [t('50–70 低い'), LOW],
                [t('< 50 とても低い'), VERY_LOW],
            ],
        },
    };
}

export const PlddtBFactorColorThemeProvider: ColorTheme.Provider<Params, 'oritatami-plddt'> = {
    name: 'oritatami-plddt',
    label: 'pLDDT (B-factor)',
    category: ColorThemeCategory.Validation,
    factory: PlddtBFactorColorTheme,
    getParams: () => Params,
    defaultValues: PD.getDefaultValues(Params),
    isApplicable: (ctx: ThemeDataContext) =>
        !!ctx.structure && ctx.structure.models.some(m => m.atomicConformation.B_iso_or_equiv.isDefined),
};
