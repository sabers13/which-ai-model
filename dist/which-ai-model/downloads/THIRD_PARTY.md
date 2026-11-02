# Third-party notices

This repository's own code and its `docs/guide/` pages are MIT licensed (see
[`LICENSE`](LICENSE)). The items below belong to someone else. Licences were read from
their sources on 2026-10-29 for the v1 items and on 2026-10-31 for the NHTSA snapshot;
the source, the retrieval date and what each item is used for sit beside it.

## Data

### BANKING77 — PolyAI (CC-BY-4.0)

- Source: <https://huggingface.co/datasets/PolyAI/banking77>
- Licence: Creative Commons Attribution 4.0 International (CC-BY-4.0),
  <https://creativecommons.org/licenses/by/4.0/>, as declared in the dataset card and
  the Hugging Face API metadata (`license:cc-by-4.0`).
- Used for: the T-ROUTE routing sample (8 messages per intent from the test split) and
  the seeded T-REPLY subset, plus the TF-IDF routing baseline trained on the train
  split. The 77 intent names are stored in `src/which_ai_model/legacy/_intents.py`
  because the legacy router needs the label vocabulary.
- What is committed: sample **ids** only (`data/samples/`); dataset text stays in the
  gitignored `data/raw/` and is fetched by `scripts/fetch_data.py`, which verifies the
  checksums in `data/CHECKSUMS.txt`.

### CLINC150 (clinc_oos, `plus` config) — CLINC (CC-BY-3.0)

- Source: <https://huggingface.co/datasets/clinc/clinc_oos>
- Licence: Creative Commons Attribution 3.0 Unported (CC-BY-3.0),
  <https://creativecommons.org/licenses/by/3.0/>, as declared in the dataset card and
  the Hugging Face API metadata (`license:cc-by-3.0`).
- Used for: the T-SCOPE out-of-scope sample (150 `oos` plus 150 in-scope messages from
  the `plus` test split) and the TF-IDF scope baseline trained on `plus` train with
  `oos` as a class.
- What is committed: sample **ids** only.

### NHTSA ODI complaints and recalls — U.S. Department of Transportation (U.S. government work, public domain)

- Source: the ODI flat files under <https://static.nhtsa.gov/odi/ffdd/>, read from the
  point-in-time snapshot of **2026-10-31**:
  - complaints <https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2020-2024.zip>
    and <https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2025-2026.zip>,
    layout <https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt>;
  - recalls <https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip>, layout
    <https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt>.
  - the five pinned sha256 values, sizes and dates are in
    `data/defect/nhtsa-spec.toml`, copied byte for byte from the data scout's snapshot.
- Licence: a U.S. government work in the public domain, with redistribution of derived
  extracts allowed. NHTSA's site policy says, in the *Ownership* section of
  <https://www.nhtsa.gov/about-nhtsa/privacy-policy> (retrieved 2026-10-31): "Information
  presented on this web site is considered public information and may be distributed or
  copied." The data.gov catalogue entries for the two datasets —
  <https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-complaints>
  (identifier `DOT-81`) and
  <https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-recalls>
  (identifier `DOT-83`) — declare `accessLevel = public` and the public-domain licence
  <http://www.usa.gov/publicdomain/label/1.0/>, which resolves to
  <https://www.usa.gov/government-copyright>. The data.gov fields fetched on 2026-11-01
  say `accessLevel public`, `identifier DOT-81`/`DOT-83` and that licence URI; the
  older "Public: This dataset is intended for public access and use. License: us-pd"
  sentence is not on the current pages, so it is not quoted as current wording.
- Privacy caveat, quoted from the same licence page: "On federal websites, other people
  may have rights to the work itself or how people use it. This includes publicity or
  privacy rights under state law." Complaints are contributed by the public, so the
  derived files drop the identity and location columns (`CITY`, `STATE`, `VIN`,
  `STATE_OF_INCIDENT`, `VEHICLE_OPERATOR`, `DEALER_NAME`, `DEALER_TEL`, `DEALER_CITY`,
  `DEALER_STATE`, `DEALER_ZIP`) and discard, never redact in place, any narrative that
  matches the spec's VIN, e-mail, phone or street-address pattern.
- Used for: the T-DEFECT component-group and safety-flag task (600-item sample) and the
  TF-IDF training set, both rebuilt by `scripts/fetch_nhtsa.py` and
  `python -m which_ai_model.bench.nhtsa build`.
- What is committed: the derived extracts `data/defect/test.jsonl` and
  `data/defect/train.jsonl.gz`, with their counts in `data/defect/STATS.md` and their
  checksums in `data/defect/CHECKSUMS.txt`. The raw flat files stay outside the
  repository (a local snapshot folder or the gitignored `data/raw/nhtsa/`).

## Embedding model

### BAAI/bge-small-en-v1.5 — BAAI (MIT)

- Source: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- Licence: MIT, <https://opensource.org/license/mit>, as declared in the model card and
  the Hugging Face API metadata (`license:mit`). The pinned revision is
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (384 dimensions), recorded in
  `src/which_ai_model/rag/embed.py`.
- Used for: CPU embeddings of the `docs/guide/` corpus and of each T-DOCS question, for
  the pgvector index and the deterministic in-memory retrieval twin used by replay.

## Frameworks and services

### LangChain (MIT)

- Source: <https://github.com/langchain-ai/langchain>
- Licence: MIT License, <https://github.com/langchain-ai/langchain/blob/master/LICENSE>
  (GitHub licence API: `MIT`).
- Used for: the model layer (`init` of provider chat models), structured output with the
  `function_calling` strategy, `langchain.agents.create_agent`, the `langchain.mcp`
  client adapter, and message/tool types. Pinned as `langchain[mcp]==1.4.7` with the
  provider integrations in `pyproject.toml`.

### LangGraph (MIT)

- Source: <https://github.com/langchain-ai/langgraph>
- Licence: MIT License, <https://github.com/langchain-ai/langgraph/blob/main/LICENSE>
  (GitHub licence API: `MIT`).
- Used for: the branching scope-check flow, the retrieve → answer → cite docs flow and
  the agent loop's compiled graph. Pinned as `langgraph==1.2.18`.

### MCP Python SDK (MIT)

- Source: <https://github.com/modelcontextprotocol/python-sdk>
- Licence: MIT License,
  <https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE> (GitHub licence
  API: `MIT`).
- Used for: the local stdio MCP server (`tools/mcp_server.py`, `MCPServer`) that
  publishes the fee-table and policy tools. Pinned as `mcp==2.3.0`. The archived
  `langchain-mcp-adapters` package is deliberately not a dependency.

### pgvector (PostgreSQL licence)

- Source: <https://github.com/pgvector/pgvector>
- Licence: the PostgreSQL License (also called the PostgreSQL licence or the
  PostgreSQL/UC licence), reproduced in
  <https://github.com/pgvector/pgvector/blob/master/LICENSE>: "Permission to use, copy,
  modify, and distribute this software and its documentation for any purpose, without
  fee, and without a written agreement is hereby granted…". GitHub's licence API reports
  it as `NOASSERTION` because the text is the PostgreSQL licence rather than an SPDX
  template.
- Used for: the `vector` column type, cosine distance operator and HNSW/IVFFlat indexes
  in the `wam_chunks` table. The Python client is the MIT-licensed `pgvector==0.5.1`
  package. The database image is `pgvector/pgvector:pg17-trixie` from `compose.yaml`.

### PostgreSQL (PostgreSQL License)

- Source: <https://www.postgresql.org/> and
  <https://github.com/postgres/postgres/blob/master/COPYRIGHT>
- Licence: the PostgreSQL License, reproduced in the `COPYRIGHT` file above ("Portions
  Copyright (c) 1996-2026, PostgreSQL Global Development Group… Permission to use,
  copy, modify, and distribute this software and its documentation for any purpose,
  without fee, and without a written agreement is hereby granted…").
- Used for: the local database that stores the guide chunks and answers similarity
  queries. It is a service dependency, run from the official pgvector image for local
  use; no PostgreSQL source is vendored here.

## Adapted source

### SaberOps — hand-written tool loop (Apache-2.0)

`src/which_ai_model/legacy/l5_agent.py` adapts the hand-written tool loop from SaberOps
`src/saberops/workers/api_agent.py` at commit
`72e04bdf63035cc33e22a6a3d5f80d2f7f9d3047`
(<https://github.com/sabers13/SaberOps/blob/72e04bd/src/saberops/workers/api_agent.py>),
Copyright Saber Sojudi, licensed under the Apache License, Version 2.0 (the repository's
`LICENSE`, confirmed through the GitHub licence API on 2026-10-29). SaberOps is earlier
work by the same author.

The adapted parts are the manual `tools` JSON, the hard turn limit, the no-redirect
policy and the assistant/tool message sequence. The original coding tools (`read_file`,
`list_files`, `write_file`, `run_command`) are **not** carried over; they are replaced by
`get_fee` and `get_policy`, which read the fictional fixtures in
`tests/fixtures/legacy/`. No SaberOps code is vendored as a package.

Because this repository distributes an adapted Apache-2.0 file, the licence text is
reproduced in full below.

---

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
