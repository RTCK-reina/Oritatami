import { StructureElement, StructureProperties, StructureSelection, type Structure } from 'molstar/lib/mol-model/structure';
import type { Loci } from 'molstar/lib/mol-model/loci';
import { EmptyLoci } from 'molstar/lib/mol-model/loci';
import type { StateObjectSelector } from 'molstar/lib/mol-state';
import type { PluginStateObject } from 'molstar/lib/mol-plugin-state/objects';
import { PluginContext } from 'molstar/lib/mol-plugin/context';
import { t } from '../i18n';
import { DefaultPluginSpec } from 'molstar/lib/mol-plugin/spec';
import { PluginConfig } from 'molstar/lib/mol-plugin/config';
import { PluginCommands } from 'molstar/lib/mol-plugin/commands';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Script } from 'molstar/lib/mol-script/script';
import type { Expression } from 'molstar/lib/mol-script/language/expression';
import { lociLabel } from 'molstar/lib/mol-theme/label';
import type { ColorTheme } from 'molstar/lib/mol-theme/color';
import { Asset } from 'molstar/lib/mol-util/assets';
import { Color } from 'molstar/lib/mol-util/color';
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra';
import { PlddtBFactorColorThemeProvider } from './plddtTheme';

export type RepStyle = 'cartoon' | 'surface' | 'ball-and-stick' | 'spacefill' | 'putty';
export type ColorMode = 'plddt' | 'chain' | 'rainbow' | 'hydrophobicity' | 'uniform' | 'secondary';

export interface LoadItem {
    url: string;
    format: 'mmcif' | 'pdb';
    label: string;
    color: ColorMode;
    /** used when color === 'uniform' */
    uniformColor?: number;
    highlights?: { chain: string; residues: number[] }[];
}

export interface ResidueRef {
    structureIndex: number;
    chain: string;
    labelSeqId: number;
    authSeqId: number;
    compId: string;
    isPolymer: boolean;
}

type StructureRef = StateObjectSelector<PluginStateObject.Molecule.Structure>;

interface Loaded {
    item: LoadItem;
    structure: StructureRef;
}

const BUILTIN_COLOR: Record<Exclude<ColorMode, 'uniform' | 'plddt'>, ColorTheme.BuiltIn> = {
    chain: 'chain-id',
    rainbow: 'sequence-id',
    hydrophobicity: 'hydrophobicity',
    secondary: 'secondary-structure',
};

type PolymerRepType = 'cartoon' | 'molecular-surface' | 'putty' | 'ball-and-stick' | 'spacefill';

const HIGHLIGHT_COLOR = 0xff3df2;

export class Viewer {
    plugin: PluginContext | null = null;
    private loaded: Loaded[] = [];
    private style: RepStyle = 'cartoon';
    private showPocket = true;
    private spinning = false;
    private clickListeners = new Set<(r: ResidueRef | null) => void>();
    private hoverListeners = new Set<(label: string | null) => void>();
    private errorListeners = new Set<(message: string) => void>();

    async mount(target: HTMLDivElement, background: number): Promise<void> {
        const spec = DefaultPluginSpec();
        spec.config = [
            ...(spec.config ?? []),
            [PluginConfig.Viewport.ShowExpand, false],
            [PluginConfig.Viewport.ShowControls, false],
            [PluginConfig.Viewport.ShowSettings, false],
            [PluginConfig.Viewport.ShowSelectionMode, false],
            [PluginConfig.Viewport.ShowAnimation, false],
        ];
        const plugin = new PluginContext(spec);
        await plugin.init();
        const ok = await plugin.mountAsync(target, { checkeredCanvasBackground: false });
        if (!ok) throw new Error(t('WebGL を初期化できませんでした'));
        plugin.representation.structure.themes.colorThemeRegistry.add(PlddtBFactorColorThemeProvider);
        await PluginCommands.Canvas3D.SetSettings(plugin, {
            settings: props => {
                props.renderer.backgroundColor = Color(background);
                props.camera.helper.axes.name = 'off';
            },
        });
        plugin.events.log.subscribe(entry => {
            if (entry.type === 'error') {
                console.error('[Mol*]', entry.message);
                this.errorListeners.forEach(fn => fn(entry.message));
            }
        });
        plugin.behaviors.interaction.click.subscribe(({ current }) => {
            const ref = this.residueFromLoci(current.loci);
            this.clickListeners.forEach(fn => fn(ref));
        });
        plugin.behaviors.interaction.hover.subscribe(({ current }) => {
            const label = StructureElement.Loci.is(current.loci) ? lociLabel(current.loci, { htmlStyling: false }) : null;
            this.hoverListeners.forEach(fn => fn(label));
        });
        this.plugin = plugin;
        (window as Window & { __oritatamiViewer?: Viewer }).__oritatamiViewer = this;
    }

    dispose(): void {
        this.plugin?.dispose();
        this.plugin = null;
    }

    onResidueClick(fn: (r: ResidueRef | null) => void): () => void {
        this.clickListeners.add(fn);
        return () => this.clickListeners.delete(fn);
    }

    onError(fn: (message: string) => void): () => void {
        this.errorListeners.add(fn);
        return () => this.errorListeners.delete(fn);
    }

    onHover(fn: (label: string | null) => void): () => void {
        this.hoverListeners.add(fn);
        return () => this.hoverListeners.delete(fn);
    }

    private need(): PluginContext {
        if (!this.plugin) throw new Error('viewer not mounted');
        return this.plugin;
    }

    private residueFromLoci(loci: Loci): ResidueRef | null {
        if (!StructureElement.Loci.is(loci)) return null;
        const loc = StructureElement.Loci.getFirstLocation(loci);
        if (!loc) return null;
        const structureIndex = this.loaded.findIndex(l => l.structure.cell?.obj?.data === loci.structure
            || l.structure.cell?.obj?.data.root === loci.structure.root);
        const entityType = StructureProperties.entity.type(loc);
        return {
            structureIndex,
            chain: StructureProperties.chain.auth_asym_id(loc),
            labelSeqId: StructureProperties.residue.label_seq_id(loc),
            authSeqId: StructureProperties.residue.auth_seq_id(loc),
            compId: StructureProperties.atom.label_comp_id(loc),
            isPolymer: entityType === 'polymer',
        };
    }

    async load(items: LoadItem[], { keepCamera = false }: { keepCamera?: boolean } = {}): Promise<void> {
        const plugin = this.need();
        const snapshot = keepCamera ? plugin.canvas3d?.camera.getSnapshot() : undefined;
        await plugin.clear();
        // plugin.clear() drops the state tree but not the downloaded files behind it. Stepping
        // through a long job list kept every mmCIF ever opened in the asset cache, so the
        // renderer process grew all session. Nothing here is read again: a re-open re-downloads
        // from the local API.
        plugin.managers.asset.clear();
        this.loaded = [];
        for (const item of items) {
            const data = await plugin.builders.data.download({ url: Asset.Url(item.url), isBinary: false, label: item.label },
                { state: { isGhost: true } });
            const trajectory = await plugin.builders.structure.parseTrajectory(data, item.format);
            const model = await plugin.builders.structure.createModel(trajectory);
            const structure = await plugin.builders.structure.createStructure(model);
            this.loaded.push({ item, structure });
        }
        await this.rebuild();
        // Commit synchronously so the bounding sphere is known before the camera moves (rAF-driven
        // commits are throttled while the window is in the background).
        plugin.canvas3d?.commit(true);
        if (snapshot) {
            plugin.canvas3d?.camera.setState(snapshot, 0);
        } else {
            PluginCommands.Camera.Reset(plugin, {});
        }
    }

    async clear(): Promise<void> {
        const plugin = this.need();
        this.loaded = [];
        await plugin.clear();
        plugin.managers.asset.clear();
    }

    get items(): LoadItem[] {
        return this.loaded.map(l => l.item);
    }

    async setStyle(style: RepStyle): Promise<void> {
        this.style = style;
        await this.rebuild();
    }

    async setColor(mode: ColorMode): Promise<void> {
        this.loaded.forEach(l => {
            if (l.item.color !== 'uniform') l.item.color = mode;
        });
        await this.rebuild();
    }

    async setHighlights(index: number, highlights: { chain: string; residues: number[] }[]): Promise<void> {
        if (!this.loaded[index]) return;
        this.loaded[index].item.highlights = highlights;
        await this.rebuild();
    }

    async setShowPocket(show: boolean): Promise<void> {
        this.showPocket = show;
        await this.rebuild();
    }

    private async polymerRep(comp: StructureRef, type: PolymerRepType, item: LoadItem,
        typeParams: { alpha?: number; ignoreHydrogens?: boolean } = {}): Promise<void> {
        const plugin = this.need();
        const rep = plugin.builders.structure.representation;
        if (item.color === 'plddt') {
            // custom theme: pass providers (the name-based overload only knows built-in themes)
            const provider = plugin.representation.structure.registry.get(type);
            await rep.addRepresentation(comp, { type: provider, typeParams, color: PlddtBFactorColorThemeProvider });
        } else if (item.color === 'uniform') {
            await rep.addRepresentation(comp, { type, typeParams, color: 'uniform', colorParams: { value: Color(item.uniformColor ?? 0x58d5c9) } });
        } else {
            await rep.addRepresentation(comp, { type, typeParams, color: BUILTIN_COLOR[item.color] });
        }
    }

    private async rebuild(): Promise<void> {
        const plugin = this.need();
        await plugin.dataTransaction(async () => {
            for (const { item, structure } of this.loaded) {
                // remove previous components of this structure
                const children = plugin.state.data.tree.children.get(structure.ref);
                const update = plugin.state.data.build();
                children.forEach(child => {
                    update.delete(child);
                });
                await update.commit();
                await this.buildStructure(structure, item);
            }
        }, { rethrowErrors: true });
        plugin.canvas3d?.commit(true);
    }

    private async component(structure: StructureRef, expr: Expression, key: string) {
        return this.need().builders.structure.tryCreateComponentFromExpression(structure, expr, key);
    }

    private async buildStructure(structure: StructureRef, item: LoadItem): Promise<void> {
        const plugin = this.need();
        const builders = plugin.builders.structure;
        const polymer = await builders.tryCreateComponentStatic(structure, 'polymer', { label: `${item.label} polymer` });
        if (polymer) {
            if (this.style === 'surface') {
                await this.polymerRep(polymer, 'cartoon', item);
                await this.polymerRep(polymer, 'molecular-surface', item, { alpha: 0.55, ignoreHydrogens: true });
            } else if (this.style === 'putty') {
                await this.polymerRep(polymer, 'putty', item);
            } else if (this.style === 'ball-and-stick' || this.style === 'spacefill') {
                await this.polymerRep(polymer, this.style, item, { ignoreHydrogens: true });
            } else {
                await this.polymerRep(polymer, 'cartoon', item);
            }
        }
        const ligand = await builders.tryCreateComponentStatic(structure, 'ligand', { label: `${item.label} ligand` });
        if (ligand) {
            await builders.representation.addRepresentation(ligand, {
                type: 'ball-and-stick', color: 'element-symbol',
                colorParams: item.color === 'uniform' ? { carbonColor: { name: 'uniform', params: { value: Color(item.uniformColor ?? 0x58d5c9) } } } : undefined,
                typeParams: { sizeFactor: 0.3 },
            });
        }
        const ion = await builders.tryCreateComponentStatic(structure, 'ion', { label: `${item.label} ion` });
        if (ion) await builders.representation.addRepresentation(ion, { type: 'ball-and-stick', color: 'element-symbol' });
        const branched = await builders.tryCreateComponentStatic(structure, 'branched', { label: `${item.label} glycan` });
        if (branched) await builders.representation.addRepresentation(branched, { type: 'carbohydrate' });

        if (this.showPocket && (ligand || ion)) {
            const pocketExpr = MS.struct.modifier.exceptBy({
                0: MS.struct.modifier.includeSurroundings({
                    0: MS.struct.generator.atomGroups({
                        'entity-test': MS.core.rel.eq([MS.ammp('entityType'), 'non-polymer']),
                    }),
                    radius: 4.5,
                    'as-whole-residues': true,
                }),
                by: MS.struct.generator.atomGroups({ 'entity-test': MS.core.rel.eq([MS.ammp('entityType'), 'non-polymer']) }),
            });
            const pocket = await this.component(structure, pocketExpr, `pocket`);
            if (pocket) {
                await builders.representation.addRepresentation(pocket, {
                    type: 'ball-and-stick', color: 'element-symbol', typeParams: { sizeFactor: 0.16, ignoreHydrogens: true },
                });
            }
        }

        const hl = (item.highlights ?? []).filter(h => h.residues.length);
        if (hl.length) {
            const expr = MS.struct.combinator.merge(hl.map(h => MS.struct.generator.atomGroups({
                'chain-test': MS.core.rel.eq([MS.struct.atomProperty.macromolecular.auth_asym_id(), h.chain]),
                'residue-test': MS.core.set.has([MS.set(...h.residues), MS.struct.atomProperty.macromolecular.label_seq_id()]),
            })));
            const comp = await this.component(structure, expr, 'highlight');
            if (comp) {
                await builders.representation.addRepresentation(comp, {
                    type: 'ball-and-stick', color: 'uniform', colorParams: { value: Color(HIGHLIGHT_COLOR) },
                    typeParams: { sizeFactor: 0.28, ignoreHydrogens: true },
                });
                await builders.representation.addRepresentation(comp, {
                    type: 'label', typeParams: { level: 'residue', textColor: Color(0xffffff), borderWidth: 0.3, textSize: 0.9 },
                });
            }
        }
    }

    private residueLoci(index: number, chain: string, labelSeqIds: number[]): Loci {
        const data: Structure | undefined = this.loaded[index]?.structure.cell?.obj?.data;
        if (!data) return EmptyLoci;
        const sel = Script.getStructureSelection(Q => Q.struct.generator.atomGroups({
            'chain-test': Q.core.rel.eq([Q.struct.atomProperty.macromolecular.auth_asym_id(), chain]),
            'residue-test': Q.core.set.has([Q.set(...labelSeqIds), Q.struct.atomProperty.macromolecular.label_seq_id()]),
            'group-by': Q.struct.atomProperty.macromolecular.residueKey(),
        }), data);
        return StructureSelection.toLociWithSourceUnits(sel);
    }

    highlightResidues(index: number, chain: string, labelSeqIds: number[]): void {
        const plugin = this.plugin;
        if (!plugin) return;
        const loci = labelSeqIds.length ? this.residueLoci(index, chain, labelSeqIds) : EmptyLoci;
        plugin.managers.interactivity.lociHighlights.highlightOnly({ loci });
    }

    focusResidues(index: number, chain: string, labelSeqIds: number[]): void {
        const plugin = this.plugin;
        if (!plugin || !labelSeqIds.length) return;
        const loci = this.residueLoci(index, chain, labelSeqIds);
        if (StructureElement.Loci.is(loci) && !StructureElement.Loci.isEmpty(loci)) {
            plugin.managers.camera.focusLoci(loci, { extraRadius: 6 });
        }
    }

    resetCamera(): void {
        if (this.plugin) PluginCommands.Camera.Reset(this.plugin, {});
    }

    async toggleSpin(): Promise<boolean> {
        return this.setSpin(!this.spinning);
    }

    async setSpin(on: boolean): Promise<boolean> {
        const plugin = this.need();
        this.spinning = on;
        const trackball = plugin.canvas3d?.props.trackball;
        if (!trackball) return false;
        await PluginCommands.Canvas3D.SetSettings(plugin, {
            settings: {
                trackball: {
                    ...trackball,
                    animate: this.spinning
                        ? { name: 'spin', params: { speed: 0.12, axis: Vec3.create(0, -1, 0) } }
                        : { name: 'off', params: {} },
                },
            },
        });
        return this.spinning;
    }

    async screenshot(opts: { resolution?: 'viewport' | 'full-hd' | 'ultra-hd'; transparent?: boolean } = {}): Promise<string> {
        const helper = this.need().helpers.viewportScreenshot;
        if (!helper) throw new Error(t('スクリーンショット機能を利用できません'));
        const previous = helper.values;
        helper.behaviors.values.next({
            ...previous,
            resolution: { name: opts.resolution ?? 'viewport', params: {} },
            transparent: opts.transparent ?? false,
        });
        try {
            return await helper.getImageDataUri();
        } finally {
            helper.behaviors.values.next(previous);
        }
    }

    async setBackground(color: number): Promise<void> {
        const plugin = this.plugin;
        if (!plugin?.canvas3d) return;
        await PluginCommands.Canvas3D.SetSettings(plugin, {
            settings: props => {
                props.renderer.backgroundColor = Color(color);
            },
        });
    }

    /** Call when the host element changes size without a window resize (splitters, fullscreen). */
    handleResize(): void {
        this.plugin?.handleResize();
    }
}

