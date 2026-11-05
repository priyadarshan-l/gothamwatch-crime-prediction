"""
Pillar 1: Semantic Web Knowledge Graph & Automated OWL/Rule Reasoner (rdflib + SPARQL)
Unifies the GothamWatch Underworld Ontology with real Chicago Crime Dataset incidents.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

GW = Namespace("http://gothamwatch.io/ontology#")


class GothamSemanticEngine:
    def __init__(self, chicago_df: Optional[pd.DataFrame] = None):
        self.graph = Graph()
        self.graph.bind("gw", GW)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.inferred_triples: List[tuple] = []
        self._build_tbox_ontology()
        self._build_abox_seed()
        if chicago_df is not None and len(chicago_df) > 0:
            self.ingest_chicago_incidents(chicago_df, max_records=35)

    def _build_tbox_ontology(self):
        """Define OWL Classes and Properties (TBox)."""
        classes = [
            "SyndicateCell", "Operative", "Vehicle", "Weapon", "ModusOperandi",
            "HighValueTarget", "CriticalInfrastructure", "Hospital", "PowerGridCore",
            "CitySector", "EscapeCorridor", "InterceptorUnit", "StreetAnomaly",
            "DecoyAlert", "LinkedSyndicateSeries", "ChicagoCrimeIncident",
        ]
        for cls_name in classes:
            self.graph.add((GW[cls_name], RDF.type, OWL.Class))

        self.graph.add((GW.Hospital, RDFS.subClassOf, GW.CriticalInfrastructure))
        self.graph.add((GW.PowerGridCore, RDFS.subClassOf, GW.CriticalInfrastructure))
        self.graph.add((GW.DecoyAlert, RDFS.subClassOf, GW.StreetAnomaly))

        obj_props = [
            "belongsToSyndicate", "usesMO", "plansTarget", "drivesVehicle",
            "armedWith", "locatedIn", "adjacentTo", "connects", "hasEscapeCorridor",
            "patrolsSector", "occurredIn", "matchesSignature", "attributedTo",
            "decoyFor", "inferredTarget", "escapeVia",
        ]
        for prop in obj_props:
            self.graph.add((GW[prop], RDF.type, OWL.ObjectProperty))

    def _build_abox_seed(self):
        """Populate Underworld Knowledge Graph Individuals (ABox)."""
        syndicates = [
            ("GhostCell", "Ghost-Cell", 0.91, ["MO_GridBlackout", "MO_VaultBreach"], "BiotechVault"),
            ("IronMaw", "Iron Maw Crew", 0.78, ["MO_ConvoyAmbush"], "ArmoredConvoy"),
            ("SableHand", "Sable Hand", 0.66, ["MO_ContainerSmuggle", "MO_VaultBreach"], "CentralVault"),
        ]
        for sid, name, threat, mos, target in syndicates:
            u = GW[sid]
            self.graph.add((u, RDF.type, GW.SyndicateCell))
            self.graph.add((u, GW.name, Literal(name, datatype=XSD.string)))
            self.graph.add((u, GW.threatIndex, Literal(threat, datatype=XSD.float)))
            self.graph.add((u, GW.plansTarget, GW[target]))
            for m in mos:
                self.graph.add((u, GW.usesMO, GW[m]))

        operatives = [
            ("O_771", "Wraith", "GhostCell", "V_28", "AR_15"),
            ("O_772", "Static", "GhostCell", "V_28", "EMP_Rig"),
            ("O_773", "Halcyon", "GhostCell", "V_31", "ThermalLance"),
            ("O_614", "Anvil", "IronMaw", "V_12", "C4"),
            ("O_615", "Rook", "IronMaw", "V_44", "Shotgun"),
            ("O_902", "Lantern", "SableHand", "V_55", "AR_15"),
            ("O_903", "Vesper", "SableHand", "V_12", "ThermalLance"),
        ]
        for oid, alias, syn, veh, wpn in operatives:
            u = GW[oid]
            self.graph.add((u, RDF.type, GW.Operative))
            self.graph.add((u, GW.alias, Literal(alias, datatype=XSD.string)))
            self.graph.add((u, GW.belongsToSyndicate, GW[syn]))
            self.graph.add((u, GW.drivesVehicle, GW[veh]))
            self.graph.add((u, GW.armedWith, GW[wpn]))

        vehicles = [
            ("V_28", "Black sedan · stolen plates"),
            ("V_31", "Armored panel van"),
            ("V_12", "Box truck · cloned livery"),
            ("V_44", "Sport motorcycle"),
            ("V_55", "Rigid-hull speedboat"),
        ]
        for vid, name in vehicles:
            self.graph.add((GW[vid], RDF.type, GW.Vehicle))
            self.graph.add((GW[vid], GW.name, Literal(name, datatype=XSD.string)))

        weapons = [
            ("AR_15", "AR-15 rifle"),
            ("EMP_Rig", "Portable EMP rig"),
            ("ThermalLance", "Thermal lance"),
            ("C4", "C4 charges"),
            ("Shotgun", "12-gauge shotgun"),
        ]
        for wid, name in weapons:
            self.graph.add((GW[wid], RDF.type, GW.Weapon))
            self.graph.add((GW[wid], GW.name, Literal(name, datatype=XSD.string)))

        mos = [
            ("MO_GridBlackout", "Grid blackout", "EMP substation spike + encrypted radio burst"),
            ("MO_VaultBreach", "Vault breach", "Thermal lance residue + seismic micro-tremor"),
            ("MO_ConvoyAmbush", "Convoy ambush", "Spike-strip deployment + GPS jamming"),
            ("MO_ContainerSmuggle", "Container smuggle", "Manifest tampering + crane override"),
        ]
        for mid, name, sig in mos:
            self.graph.add((GW[mid], RDF.type, GW.ModusOperandi))
            self.graph.add((GW[mid], GW.name, Literal(name, datatype=XSD.string)))
            self.graph.add((GW[mid], GW.signature, Literal(sig, datatype=XSD.string)))

        targets = [
            ("CentralVault", "Gotham Central Vault", "S1", 0.95),
            ("BiotechVault", "Wayne Biotech Cold Vault", "S6", 0.97),
            ("DiamondExchange", "Diamond Exchange", "S4", 0.82),
            ("ArmoredConvoy", "Armored Transit Convoy", "S1", 0.88),
        ]
        for tid, name, sec, val in targets:
            self.graph.add((GW[tid], RDF.type, GW.HighValueTarget))
            self.graph.add((GW[tid], GW.name, Literal(name, datatype=XSD.string)))
            self.graph.add((GW[tid], GW.locatedIn, GW[sec]))
            self.graph.add((GW[tid], GW.assetValue, Literal(val, datatype=XSD.float)))

        infra = [
            ("GothamGeneral", "Gotham General Hospital", "Hospital", "S5"),
            ("MercyHospital", "Mercy Hospital", "Hospital", "S7"),
            ("NarrowsSubstation", "Narrows Power Substation", "PowerGridCore", "S3"),
        ]
        for iid, name, cls, sec in infra:
            self.graph.add((GW[iid], RDF.type, GW[cls]))
            self.graph.add((GW[iid], RDF.type, GW.CriticalInfrastructure))
            self.graph.add((GW[iid], GW.name, Literal(name, datatype=XSD.string)))
            self.graph.add((GW[iid], GW.locatedIn, GW[sec]))

        sectors = [
            ("S1", "Financial Core", ["S3", "S4", "S5", "S6", "S7"]),
            ("S2", "Industrial Docks", ["S3", "S5"]),
            ("S3", "The Narrows", ["S1", "S2", "S4", "S5"]),
            ("S4", "Diamond District", ["S1", "S3", "S6"]),
            ("S5", "Old Gotham", ["S1", "S2", "S3", "S7"]),
            ("S6", "Wayne Biotech", ["S1", "S4", "S7"]),
            ("S7", "East End", ["S1", "S5", "S6", "S8"]),
            ("S8", "Arkham Causeway", ["S7"]),
        ]
        for sid, name, adj in sectors:
            self.graph.add((GW[sid], RDF.type, GW.CitySector))
            self.graph.add((GW[sid], GW.name, Literal(name, datatype=XSD.string)))
            for a in adj:
                self.graph.add((GW[sid], GW.adjacentTo, GW[a]))

        bridges = [
            ("R_1", "Gotham Bridge", "S4", "S6"),
            ("R_2", "North Bridge", "S1", "S6"),
            ("R_3", "Westward Bridge", "S1", "S7"),
            ("R_4", "Sprang Bridge", "S5", "S7"),
        ]
        for bid, name, s_a, s_b in bridges:
            self.graph.add((GW[bid], RDF.type, GW.EscapeCorridor))
            self.graph.add((GW[bid], GW.name, Literal(name, datatype=XSD.string)))
            self.graph.add((GW[bid], GW.connects, GW[s_a]))
            self.graph.add((GW[bid], GW.connects, GW[s_b]))
            self.graph.add((GW[s_a], GW.hasEscapeCorridor, GW[bid]))
            self.graph.add((GW[s_b], GW.hasEscapeCorridor, GW[bid]))

        units = [
            ("IC_01", "S3"), ("IC_02", "S4"), ("IC_03", "S2"), ("IC_04", "S5"),
            ("IC_05", "S1"), ("IC_06", "S6"), ("IC_07", "S7"), ("IC_08", "S8"),
            ("IC_09", "S7"), ("IC_10", "S5"), ("IC_11", "S6"), ("IC_12", "S3"),
        ]
        for uid, sec in units:
            self.graph.add((GW[uid], RDF.type, GW.InterceptorUnit))
            self.graph.add((GW[uid], GW.patrolsSector, GW[sec]))
            self.graph.add((GW[uid], GW.status, Literal("READY", datatype=XSD.string)))

    def ingest_chicago_incidents(self, df: pd.DataFrame, max_records: int = 35):
        """Link real high-severity Chicago Crime incidents into the RDF Knowledge Graph."""
        high_df = df[df["is_syndicate_signature"] == 1].head(max_records)
        type_to_mo = {
            "ROBBERY": "MO_ConvoyAmbush",
            "BURGLARY": "MO_VaultBreach",
            "WEAPONS VIOLATION": "MO_ConvoyAmbush",
            "ARSON": "MO_GridBlackout",
            "MOTOR VEHICLE THEFT": "MO_ContainerSmuggle",
        }
        for _, row in high_df.iterrows():
            case_no = str(row.get("case_number", row.get("id", "CASE"))).strip()
            inc_uri = GW[f"CHI_{case_no}"]
            sec_uri = GW[str(row["sector"])]
            ptype = str(row["primary_type"]).upper()
            mo_id = type_to_mo.get(ptype, "MO_VaultBreach")

            self.graph.add((inc_uri, RDF.type, GW.ChicagoCrimeIncident))
            self.graph.add((inc_uri, GW.caseNumber, Literal(case_no, datatype=XSD.string)))
            self.graph.add((inc_uri, GW.primaryType, Literal(ptype, datatype=XSD.string)))
            self.graph.add((inc_uri, GW.occurredIn, sec_uri))
            self.graph.add((inc_uri, GW.matchesSignature, GW[mo_id]))

    def inject_anomaly_and_infer(
        self,
        anomaly_id: str = "ANOM_202",
        sector_id: str = "S3",
        mo_id: str = "MO_GridBlackout",
        description: str = "Substation power spike + encrypted radio burst in The Narrows (S3)",
    ) -> List[tuple]:
        """
        Inject a street-level anomaly and execute OWL/Rule semantic materialization:
        Rule 1 (Syndicate Attribution): Anomaly matches MO -> Syndicate uses MO -> attributedTo Syndicate
        Rule 2 (Decoy Alert Deduction): Anomaly in lower-value sector while Syndicate plansTarget in
                high-value Vault (>0.90) -> infer DecoyAlert & inferredTarget.
        """
        anom_uri = GW[anomaly_id]
        self.graph.add((anom_uri, RDF.type, GW.StreetAnomaly))
        self.graph.add((anom_uri, GW.name, Literal(description, datatype=XSD.string)))
        self.graph.add((anom_uri, GW.occurredIn, GW[sector_id]))
        self.graph.add((anom_uri, GW.matchesSignature, GW[mo_id]))

        new_inferences = []

        # Rule 1: Syndicate Attribution via shared Modus Operandi
        q_rule1 = """
        PREFIX gw: <http://gothamwatch.io/ontology#>
        SELECT ?syndicate ?target ?targetSector ?val WHERE {
            ?anom gw:matchesSignature ?mo .
            ?syndicate a gw:SyndicateCell ;
                       gw:usesMO ?mo ;
                       gw:plansTarget ?target .
            ?target gw:locatedIn ?targetSector ;
                    gw:assetValue ?val .
            FILTER(?anom = gw:%s)
        }
        """ % anomaly_id

        for row in self.graph.query(q_rule1):
            syn_uri, target_uri, target_sec_uri, val_lit = row
            t1 = (anom_uri, GW.attributedTo, syn_uri)
            t2 = (anom_uri, RDF.type, GW.LinkedSyndicateSeries)
            self.graph.add(t1)
            self.graph.add(t2)
            new_inferences.extend([t1, t2])

            # Rule 2: Decoy Trap Deduction if anomaly sector != high-value target sector
            if str(target_sec_uri) != str(GW[sector_id]) and float(val_lit) >= 0.85:
                t3 = (anom_uri, RDF.type, GW.DecoyAlert)
                t4 = (anom_uri, GW.decoyFor, target_uri)
                t5 = (syn_uri, GW.inferredTarget, target_uri)
                for trip in (t3, t4, t5):
                    self.graph.add(trip)
                    new_inferences.append(trip)

                # Rule 3: Deduce primary escape bridge from target sector
                q_bridge = """
                PREFIX gw: <http://gothamwatch.io/ontology#>
                SELECT ?bridge WHERE {
                    <%s> gw:hasEscapeCorridor ?bridge .
                } LIMIT 1
                """ % str(target_sec_uri)
                for brow in self.graph.query(q_bridge):
                    t6 = (syn_uri, GW.escapeVia, brow[0])
                    self.graph.add(t6)
                    new_inferences.append(t6)

        self.inferred_triples.extend(new_inferences)
        return new_inferences

    def run_sparql_to_df(self, query: str) -> pd.DataFrame:
        """Execute a SPARQL SELECT query and return clean human-readable DataFrame."""
        res = self.graph.query(query)
        cols = [str(v) for v in res.vars]
        rows = []
        for r in res:
            clean_row = {}
            for c, val in zip(cols, r):
                if val is None:
                    clean_row[c] = ""
                else:
                    s = str(val)
                    if "#" in s:
                        s = "gw:" + s.split("#")[-1]
                    clean_row[c] = s
            rows.append(clean_row)
        return pd.DataFrame(rows, columns=cols)
