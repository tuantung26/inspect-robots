# Tich hop va danh gia model moi

Tai lieu nay mo ta du lieu dau vao, dau ra va luong chay khi dua mot policy/model moi vao Inspect Robots. Framework cung cap giao dien chung; adapter van phai chuyen doi du lieu theo dung yeu cau cua model va robot.

## Cac thanh phan dau vao

Mot lan danh gia ghep ba thanh phan:

| Thanh phan | Vai tro |
| --- | --- |
| `Task` | Tap cac tinh huong can thu, tieu chi cham va gioi han rollout. |
| `Policy` | Adapter goi model, chuyen quan sat thanh input model va chuyen output model thanh action. |
| `Embodiment` | Adapter robot hoac simulator, tao quan sat va thuc thi action. |

`Task` chua cac `Scene`. Moi scene co the mo ta chi dan, target, seed va metadata. So lan lap (`epochs`) giup danh gia do on dinh cua policy tren cung mot scene. Xem [`Task`](../../src/inspect_robots/task.py) va [`Scene`](../../src/inspect_robots/scene.py).

## Du lieu policy nhan

Framework goi `policy.act(observation)` voi mot [`Observation`](../../src/inspect_robots/types.py), gom cac truong chuan:

| Truong | Dinh dang |
| --- | --- |
| `images` | Mapping tu ten camera sang anh NumPy `uint8`, kich thuoc `(H, W, C)`. |
| `state` | Mapping tu ten trang thai sang mang NumPy, vi du `joint_pos`, `eef_pos` hoac `gripper`. |
| `instruction` | Chi dan van ban, co the la `None`. |
| `image_times`, `state_time` | Moc thoi gian cua du lieu neu embodiment cung cap. |
| `extra` | Du lieu bo sung; rollout tu them buoc moi truong va thong tin operator. |

Day la giao dien cua framework, khong phai dinh dang bat buoc cua model. Policy adapter tu resize/normalize anh, tao tensor, chuan hoa state hoac doi ten truong theo yeu cau model. Hop dong cua policy la [`Policy`](../../src/inspect_robots/policy.py); mo ta camera, state va action space nam trong [`spaces.py`](../../src/inspect_robots/spaces.py).

### Neu model can video tham chieu

`Observation` khong co truong video tham chieu rieng. Co the dat dinh danh hoac duong dan clip trong metadata cua scene, sau do policy doc thong tin do tai `reset(scene)` va nap/cat frame theo yeu cau model. Neu clip can duoc scorer dung lai, hay dua thong tin can thiet vao `Target.spec` hoac record metadata; scorer nhan `Record` va `Target`, khong nhan truc tiep `Scene`.

Adapter can tu xu ly dinh dang video, toc do frame, dong bo thoi gian va cach ghep clip tham chieu voi quan sat hien tai.

## Dau ra policy

Policy tra ve mot [`ActionChunk`](../../src/inspect_robots/types.py), gom mot hoac nhieu `Action`. Moi `Action.data` la mang so NumPy co so chieu phai khop voi action space cua embodiment. Y nghia cua vector do duoc khai bao rieng, vi du:

- `joint_pos`: vi tri muc tieu cua cac khop.
- `joint_vel`: van toc khop.
- `joint_delta`: thay doi vi tri khop.
- `eef_abs_pose` hoac `eef_delta_pose`: pose tuyet doi hoac thay doi cua dau cong tac.

Action space con khai bao gioi han, frame toa do, bieu dien xoay va nhan tung chieu neu can. Adapter phai chuyen output model sang dung don vi, thu tu khop, frame va control mode. Khong nen gia dinh framework co the suy ra cac y nghia nay tu vector.

Action chunk cho phep model du doan nhieu buoc tu mot lan suy luan. Action manager chon cac buoc de thuc thi va thoi diem goi policy lai. Policy co the phat action ket thuc nhu `done` hoac `give_up`, nhung day khong phai phan xet thanh cong.

## Embodiment va ket qua moi buoc

Embodiment khai bao `info`, sau do cung cap:

```python
observation = embodiment.reset(scene, seed=seed)
result = embodiment.step(action)
```

`reset()` chuan bi scene va tra ve quan sat ban dau. `step()` thuc thi mot action va tra ve `StepResult`: quan sat moi, tuy chon `reward`, co ket thuc hay khong, ly do ket thuc, va `info` tuy chon cho scorer (vi du khoang cach toi muc tieu).

Voi robot that, embodiment chiu trach nhiem ket noi, doc sensor, gui lenh, va dam bao nhip dieu khien an toan. Rollout khong tu cho theo chu ky; embodiment can tu pacing trong `step()` neu phan cung can dieu do. Xem [`Embodiment`](../../src/inspect_robots/embodiment.py).

## Pipeline danh gia

```text
Task (Scene + rubric + budget)
              |
              v
Policy va Embodiment khai bao spaces
              |
              v
Kiem tra tuong thich truoc rollout
              |
              v
Reset policy va embodiment theo scene/seed
              |
              v
Observation -> policy/model -> ActionChunk
              |
              v
Action manager -> guardrails -> embodiment.step()
              |
              +---- quan sat moi, lap lai den khi ket thuc
              |
              v
Record cua trial -> grader neu duoc cau hinh -> scorer
              |
              v
Gop diem qua epochs va scenes -> EvalLog
```

Truoc khi chay, framework kiem tra kich thuoc va ngu nghia action, cac camera/state policy can, va kha nang dap ung budget cua task. Trong rollout, guard co the gioi han action hoac dung run neu co loi an toan. Moi buoc, action chunk, su kien va loi duoc ghi vao record.

Sau rollout, grader co the ghi nhan phan xet cua operator hoac VLM. Scorer doc record va target de cham; scorer khong dieu khien model. Reducer nhu `mean` gop cac epoch cua mot scene, sau do metric tong hop tren cac scene co diem. Neu muon cham do giong video, benchmark can dinh nghia scorer va du lieu can thiet, vi du cach can chinh thoi gian va sai so quỹ dao; viec dua video vao model khong tu dong tao ra metric do. Xem [Scoring](scoring.md).

## Dau ra cua lan danh gia

`eval()` tra ve danh sach `EvalLog` va mac dinh ghi JSON vao thu muc log. Log chua:

- Ten task, policy, embodiment; seed va cau hinh chay.
- Trang thai, ly do ket thuc va loi theo scene/trial.
- Diem theo rubric, scene va epoch; metric tong hop.
- So buoc, thoi gian chay va do tre suy luan.
- Tuy chon: record chi tiet, action chunks, anh, transcript va du lieu Rerun.

Co the xem JSON bang `inspect-robots inspect LOG.json` hoac tao bao cao bang `inspect-robots view LOG.json`. Cau truc log duoc dinh nghia trong [`log.py`](../../src/inspect_robots/log.py).

## Checklist khi tich hop model moi

1. Xac dinh model can nhung dau vao nao va adapter se lay chung tu `Observation`, `Scene` hay nguon ngoai.
2. Dinh nghia action space cua embodiment: so chieu, don vi, thu tu, frame, gioi han va control mode.
3. Viet policy adapter de tien xu ly input model va chuyen output thanh `ActionChunk`.
4. Viet embodiment adapter cho simulator hoac robot; bat dau bang simulator neu co the.
5. Tao task/scenes va scorer phu hop voi muc tieu. Tach rieng cham hoan thanh, an toan va do giong demo neu can.
6. Chay kiem tra tuong thich, sau do thu voi seed co dinh va xem log truoc khi thu tren phan cung that.

Framework cung cap pipeline va giao dien chung, nhung khong tu dong viet adapter cho model, robot hoac video format moi.
